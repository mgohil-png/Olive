### this code uses onnx graph surgeon for transformations
import logging
import subprocess
import sys
from collections import defaultdict

import numpy as np
import onnx
import onnxscript
from onnx import TensorProto, helper, numpy_helper
from onnxscript import ir
from onnxscript.rewriter import pattern

logger = logging.getLogger(__name__)

"""
ALL transform function starts with transform_ and called from GraphSurgeries pass.
ASSUMPTIONS:
- quantized values are either uint8 or uint16
Hardcoded:
- Transpose perm=[0,3,2,1]
- initializers to 4D
  - 1D [K] -> [1x1x1xK]
  - 2D [CxK] -> [KxCx1x1]
  - 3D insert 1 at 3rd dimension
- Reshape-ReduceSum to Slice-ReduceSum-Concat axes
- Expand 3D to 4D. Insert 1 at 0th dimension
- output Squeeze if Squeeze node exists axes = [0, 3]
- Flatten to reshape with axes [1, 1, 1, -1]
- Unsqueeze
    - From 3D axes=[1]
    - From 2D axes=[0, -1]
- Squeeze
    - Update existing squeeze: [0, 3]
    - New squeeze: [1, 2]
- Gather
    - indices = [2]
- ReduceSum axes = [2] or keep [-1]
"""


###
# Private helper functions
###
def get_tensor_shape_map(graph_value_info):
    tensor_name_dim_map = {}
    for value_info in graph_value_info:
        tensor_type = value_info.type.tensor_type
        shape = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append(0)
                    break
        tensor_name_dim_map[value_info.name] = shape
    return tensor_name_dim_map


def get_initializer_by_name(model, init_name):
    for init in model.graph.initializer:
        if init.name == init_name:
            return numpy_helper.to_array(init)
    return None


def calculate_clip_range(node, model):
    x_scale = get_initializer_by_name(model, node.input[1])
    x_zero_point = get_initializer_by_name(model, node.input[2])
    assert x_scale, f"{node.name} should have x_scale value"
    int_max = np.int32(65535 if x_zero_point.dtype == np.uint16 else 255 if x_zero_point.dtype == np.uint8 else 127)
    int_min = np.int32(0 if x_zero_point.dtype == np.uint16 else 0 if x_zero_point.dtype == np.uint8 else -128)
    if x_zero_point is None:
        logger.info("x_zero_point is None!")
        x_zero_point = np.array(0, dtype=np.int32)
    else:
        x_zero_point = x_zero_point.astype(np.int32)
    clip_min = ((int_min - x_zero_point) * x_scale).astype(np.float32)
    clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
    return clip_min, clip_max


###
# End of private helper functions
###
###
# Start of transform_* functions
###


###
# Replace 2D Gemm/MatMul with Transpose and 1x1 Conv
#
# Modification requirement:
# When C==1, convert it to 1x1 Conv using TRANSPOSE + CONV + TRANSPOSE sequence
###
def transform_matmul_to_transpose_conv_transpose(model):
    cnt = 0
    graph = model.graph
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
    input_shape_map = get_tensor_shape_map(graph.input)
    tensor_name_dim_map.update(input_shape_map)

    # For model outputs
    output_shape_map = get_tensor_shape_map(graph.output)
    tensor_name_dim_map.update(output_shape_map)
    initializer_dim_map = {init.name: len(init.dims) for init in graph.initializer}
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == "MatMul" or node.op_type == "Gemm":
            need_transform = False

            for input_ in node.input:
                # Input is either in initializer or value_info
                if (input_ in initializer_dim_map and initializer_dim_map[input_] != 4) or (
                    input_ in tensor_name_dim_map and len(tensor_name_dim_map[input_]) != 4
                ):
                    need_transform = True
                    break
            if need_transform:
                nodes_to_remove.append(node)

    initializers_to_remove = []
    initializers_to_add = []

    for node in nodes_to_remove:
        matmul_node_name = node.name
        nodes_to_add = []
        conv_inputs = []

        # Check the first input to Gemm/MatMul.
        # If C (channel) dimesion is 1, needs to add Transposes around Conv.
        # Otherwise, no need to add Transposes for now.
        def check_to_apply_transpose(conv_node):
            initializer_names = {init.name for init in model.graph.initializer}
            bool_value = False
            if (
                conv_node.input[1] in initializer_names
            ):  #### if second input is initializer with suitable conditions, we need to trans-conv-transpose
                for init in model.graph.initializer:
                    if init.name == conv_node.input[1]:
                        shape = list(init.dims)
                        if len(shape) == 2 or (len(shape) == 3 and shape[0] == 1) or (len(shape) == 4 and shape[1] == 1):
                            bool_value = True
                if bool_value:
                    return bool_value
            else:
                input_ = conv_node.input[1]
                if input_ in tensor_name_dim_map:
                    shape = tensor_name_dim_map[input_]
                    if len(shape) == 2:
                        # if shape[1] != 1:  # The 2D tensor (MxN, width and height) will later adding leading dimensions to 4D with 1x1xMxN #### changed from != to == ans removed condition on shape[0] and removed condition on shape[1]
                        bool_value = True
                    if len(shape) == 3 and shape[0] == 1:  # C == 1
                        bool_value = True
                    if len(shape) == 4 and shape[1] == 1:  # C == 1
                        bool_value = True
                    if (
                        len(shape) == 4 and shape[2] != 1 and shape[3] != 1
                    ):  # In some cases, C != 1 but it still needs the transpose
                        bool_value = True
            if not (bool_value):
                return bool_value
            bool_value = False
            input_ = conv_node.input[0]
            if input_ in tensor_name_dim_map:
                shape = tensor_name_dim_map[input_]
                if (
                    len(shape) == 2 and shape[1] != 1
                ):  # The 2D tensor (MxN, width and height) will later adding leading dimensions to 4D with 1x1xMxN #### changed from != to == ans removed condition on shape[0]
                    bool_value = True
                if len(shape) == 3 and shape[0] == 1:  # C == 1
                    bool_value = True
                if len(shape) == 4 and shape[1] == 1:  # C == 1
                    bool_value = True
                if (
                    len(shape) == 4 and shape[2] != 1 and shape[3] != 1
                ):  # In some cases, C != 1 but it still needs the transpose
                    bool_value = True
            # Otherwise, we considered the input tensor has been "transposed" to perform 1x1 Conv
            return bool_value

        need_to_apply_transpose = True
        need_to_apply_transpose = check_to_apply_transpose(node)  #### this was commented out
        graph.node.remove(node)
        need_to_apply_first_transpose = need_to_apply_transpose
        input_shape = tensor_name_dim_map.get(node.input[0])
        if input_shape[-1] == 1 and input_shape[-2] == 1:
            need_to_apply_first_transpose = False

        # Add Transpose if needed
        if need_to_apply_first_transpose:  #### START OF block of items newly added
            initializer_names = {init.name for init in model.graph.initializer}

            # Transpose for first input (always dynamic in this context)
            transpose_before_node_0 = helper.make_node(
                "Transpose",
                inputs=[node.input[0]],
                outputs=[f"{matmul_node_name}_transpose_before_output0_{cnt}"],
                name=f"{matmul_node_name}_transpose_before0_{cnt}",
                perm=[0, 3, 2, 1],
            )
            nodes_to_add.append(transpose_before_node_0)
            conv_input0 = f"{matmul_node_name}_transpose_before_output0_{cnt}"

            # Check if second input is dynamic or initializer
            if node.input[1] in initializer_names:
                # If it's an initializer, use it directly (reshape as needed elsewhere)
                conv_input1 = node.input[1]
            else:
                # If it's dynamic, add a Transpose node
                transpose_before_node_1 = helper.make_node(
                    "Transpose",
                    inputs=[node.input[1]],
                    outputs=[f"{matmul_node_name}_transpose_before_output1_{cnt}"],
                    name=f"{matmul_node_name}_transpose_before1_{cnt}",
                    perm=[3, 2, 0, 1],
                )
                nodes_to_add.append(transpose_before_node_1)
                conv_input1 = f"{matmul_node_name}_transpose_before_output1_{cnt}"

                # Use the prepared inputs for Conv
            conv_inputs = [conv_input0, conv_input1]
        else:
            conv_inputs = [node.input[0], node.input[1]]

        if len(node.input) == 3:  # Gemm has optional third input
            conv_inputs.append(node.input[2])

        # Add Conv
        if need_to_apply_transpose:
            node_to_add = helper.make_node(
                "Conv",
                inputs=conv_inputs,
                outputs=[f"{matmul_node_name}_transpose_output_{cnt}"],
                name=f"{matmul_node_name}_conv_{cnt}",
            )
        else:
            node_to_add = (
                helper.make_node(  #### we dont want mul/gemm to get replaced by conv so added this else statement
                    node.op_type, inputs=conv_inputs, outputs=node.output, name=f"{matmul_node_name}_new"
                )
            )
        nodes_to_add.append(node_to_add)

        # Update Conv's weight to 4D if needed
        def update_initializers(graph, name, initializers_to_remove, initializers_to_add, trans_b=False):
            for initializer in graph.initializer:
                if initializer.name != name:
                    continue
                # 2D [CxK] -> [KxCx1x1] if TransB != 1
                # 2D [CxK] -> [CxKx1x1] if TransB == 1
                if len(initializer.dims) == 2:
                    c, k = initializer.dims[0], initializer.dims[1]
                    init_arr = numpy_helper.to_array(initializer)
                    if not trans_b:
                        init_arr = init_arr.T
                        shape = (k, c, 1, 1)
                    else:
                        shape = (c, k, 1, 1)
                #### Missed 3D case ####
                else:
                    c, k, m = initializer.dims[0], initializer.dims[1], initializer.dims[2]
                    init_arr = numpy_helper.to_array(initializer)
                    if not trans_b:
                        init_arr = init_arr.T
                        shape = (m, k, c, 1)
                    else:
                        shape = (c, k, m, 1)
                reshaped_arr = np.reshape(init_arr, shape)
                new_initializer = numpy_helper.from_array(reshaped_arr, initializer.name)
                initializers_to_remove.append(initializer)
                initializers_to_add.append(new_initializer)

        # Check input B of Gemm is transposed or not
        trans_b = False
        if node.op_type == "Gemm":
            for attr in node.attribute:
                if attr.name == "transB" and attr.i == 1:
                    trans_b = True
                    break

        if need_to_apply_transpose:
            update_initializers(graph, node.input[1], initializers_to_remove, initializers_to_add, trans_b)

        if need_to_apply_transpose:
            transpose_after_node = helper.make_node(
                "Transpose",
                inputs=[f"{matmul_node_name}_transpose_output_{cnt}"],
                outputs=node.output,
                name=f"{matmul_node_name}_transpose_after_{cnt}",
                perm=[0, 3, 2, 1],
            )
            nodes_to_add.append(transpose_after_node)

        graph.node.extend(nodes_to_add)
        cnt += 1
    for init in initializers_to_remove:
        for existing in list(graph.initializer):
            if existing.name == init.name:
                graph.initializer.remove(existing)
                break
    [graph.initializer.append(init) for init in initializers_to_add]

    logger.info("Replaced %d MatMul nodes with Transpose-Conv-Transpose nodes", cnt)


def transform_remove_intermediary_squeeze_and_unsqueeze(model):
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    input_names = {input_.name for input_ in graph.input}
    output_names = {output.name for output in graph.output}
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
    initializer_dim_map = {init.name: list(init.dims) for init in graph.initializer}

    def get_shape(name):
        if name in tensor_name_dim_map:
            return tensor_name_dim_map[name]
        if name in initializer_dim_map:
            return initializer_dim_map[name]
        # Check graph inputs
        for graph_input in graph.input:
            if graph_input.name == name:
                tensor_type = graph_input.type.tensor_type
                shape = []
                if tensor_type.HasField("shape"):
                    for d in tensor_type.shape.dim:
                        if d.HasField("dim_value"):
                            shape.append(d.dim_value)
                        else:
                            shape.append(0)
                return shape
        return None

    # Build output to consumers map
    output_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            output_to_consumers.setdefault(input_name, []).append(node)

    nodes_to_remove = []
    nodes_to_add = []
    input_shape = None
    for node in graph.node:
        # --- Remove intermediary Squeeze nodes ---
        if len(node.input) > 0:
            input_shape = get_shape(
                node.input[0]
            )  #### without this if condition, it will give error for constant op (GT)

        # --- Handle special case: Squeeze with axis 2 and 5D input ---
        if (
            node.op_type == "Squeeze" and input_shape is not None and len(input_shape) == 5
        ):  ############### added as squeeze was removing axis 2 and not 0...very model specific approach used here, fixing the size and axis in the condition, but this can be improved
            # Get axes from initializer
            axes = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    axes = numpy_helper.to_array(init)
                    break
            if axes is None:
                for n in graph.node:
                    if n.op_type == "Constant" and n.output[0] == node.input[1]:
                        for attr in n.attribute:
                            if attr.name == "value":
                                axes = numpy_helper.to_array(attr.t)
                                break
                        break

            # Check if axis is 2
            if axes is not None and len(axes) == 1 and axes[0] == 2:
                # Convert squeeze to reshape with shape [1, input[1], input[3], input[4]]
                reshape_shape = [1, input_shape[1], input_shape[3], input_shape[4]]

                # Create shape initializer for Reshape
                reshape_shape_name = node.name + "_reshape_shape"
                reshape_shape_init = numpy_helper.from_array(
                    np.array(reshape_shape, dtype=np.int64), reshape_shape_name
                )
                graph.initializer.append(reshape_shape_init)

                # Create Reshape node
                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[node.input[0], reshape_shape_name],
                    outputs=node.output,
                    name=node.name + "_to_reshape",
                )
                nodes_to_add.append(reshape_node)
                nodes_to_remove.append(node)

                # Rewire consumers
                squeeze_output = node.output[0]
                consumers = output_to_consumers.get(squeeze_output, [])
                for consumer in consumers:
                    for i, input_name in enumerate(consumer.input):
                        if input_name == squeeze_output:
                            consumer.input[i] = reshape_node.output[0]
                continue  # Don't process further

        #### Here we are checking if the input and output of squeeze/unsqueeze are not model inputs/outputs ####
        if (
            (node.op_type == "Squeeze" or (node.op_type == "Unsqueeze" and len(input_shape) == 4))
            and node.input[0] not in input_names
            and node.output[0] not in output_names
        ):
            squeeze_input = node.input[0]
            squeeze_output = node.output[0]
            # Rewire consumers to use the input directly
            consumers = output_to_consumers.get(squeeze_output, [])
            for consumer in consumers:
                for i, input_name in enumerate(consumer.input):
                    if input_name == squeeze_output:
                        consumer.input[i] = squeeze_input
            nodes_to_remove.append(node)
            continue  # Don't process further

        # --- Replace eligible Unsqueeze nodes with Reshape ---
        if node.op_type == "Unsqueeze" and node.input[0] not in input_names and node.output[0] not in output_names:
            input_shape = get_shape(node.input[0])
            if input_shape is None:
                continue  # Can't process if shape is unknown

            # Get axes from initializer
            axes = None
            for init in graph.initializer:
                if init.name == node.input[1]:
                    axes = numpy_helper.to_array(init)
                    break
            if axes is None:
                for n in graph.node:
                    if (
                        n.op_type == "Constant" and n.output[0] == node.input[1]
                    ):  #### GT has value stored in constant op
                        for attr in n.attribute:
                            if attr.name == "value":
                                axes = numpy_helper.to_array(attr.t)
                                break
                        break
            if axes is None:
                continue  # Can't process if axes unknown
            # Compute unsqueezed shape
            out_shape = list(input_shape)
            for axis in sorted(axes):
                out_shape.insert(axis, 1)
            # If input is 2D and output is 3D, prepend a 1
            if len(input_shape) == 2 and len(out_shape) == 3:
                out_shape = [1, *out_shape]

            # Only replace if:
            # - input is 2D and output is 4D
            # - input is 3D and output is 4D
            # - input is 2D and output is 3D (with extra 1 at front)
            if (
                (len(input_shape) == 2 and len(out_shape) == 4)
                or (len(input_shape) == 3 and len(out_shape) == 4)
                or (len(input_shape) == 2 and len(out_shape) == 3)
            ):
                # Create shape initializer for Reshape
                reshape_shape_name = node.name + "_reshape_shape"
                reshape_shape = numpy_helper.from_array(np.array(out_shape, dtype=np.int64), reshape_shape_name)
                graph.initializer.append(reshape_shape)

                # Create Reshape node
                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[node.input[0], reshape_shape_name],
                    outputs=node.output,
                    name=node.name + "_to_reshape",
                )
                nodes_to_add.append(reshape_node)
                nodes_to_remove.append(node)

                # Rewire consumers
                unsqueeze_output = node.output[0]
                consumers = output_to_consumers.get(unsqueeze_output, [])
                for consumer in consumers:
                    for i, input_name in enumerate(consumer.input):
                        if input_name == unsqueeze_output:
                            consumer.input[i] = reshape_node.output[0]

    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)
    for node in nodes_to_add:
        graph.node.append(node)

    logger.info(
        "Removed %d intermediary Squeeze ops and replaced %d Unsqueeze ops with Reshape as per rules.",
        sum(1 for n in nodes_to_remove if n.op_type == "Squeeze"),
        sum(1 for n in nodes_to_remove if n.op_type == "Unsqueeze"),
    )


def transform_qdq_to_clip(model):
    cnt = 0
    qualin_name_node_map, deqlin_name_node_map = (
        {},
        {},
    )  # quantize_output_name -> quantize_node, dequantize_input_name -> dequantize_node
    node_datatype_map = {}  # node_name -> datatype
    graph = model.graph
    clip_range = {}  # deq_input_name -> (clip_min, clip_max)
    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # check no output or if multiple connected nodes
            if not node.output[0] or len(node.output) > 1:
                continue
            qualin_name_node_map[node.output[0]] = node
            node_datatype_map[node.output[0]] = get_initializer_by_name(model, node.input[2]).dtype
        if node.op_type == "DequantizeLinear":
            if not node.input[0]:
                continue
            assert len(node.input) == 3
            deqlin_name_node_map[node.input[0]] = node
            x_scale = get_initializer_by_name(model, node.input[1])
            x_zero_point = get_initializer_by_name(model, node.input[2])
            assert x_scale, f"{node.name} should have x_scale value"
            int_max = np.int32(
                65535 if x_zero_point.dtype == np.uint16 else 255 if x_zero_point.dtype == np.uint8 else 127
            )
            int_min = np.int32(0 if x_zero_point.dtype == np.uint16 else 0 if x_zero_point.dtype == np.uint8 else -128)
            if x_zero_point is None:
                logger.info("x_zero_point is None!")
                x_zero_point = np.array(0, dtype=np.int32)
            else:
                x_zero_point = x_zero_point.astype(np.int32)
            clip_min = ((int_min - x_zero_point) * x_scale).astype(np.float32)
            clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
            clip_range[node.input[0]] = (clip_min, clip_max)
    subgraph_output_input_map = {}  # deqlin output -> qualin input
    clip_nodes_to_add = []
    for q_output, qualin_node in qualin_name_node_map.items():
        if q_output in deqlin_name_node_map and (node_datatype_map[q_output] in ["uint16", "int16"]):
            deqlin_node = deqlin_name_node_map[q_output]
            graph.node.remove(qualin_node)
            graph.node.remove(deqlin_node)
            subgraph_output_input_map[deqlin_node.output[0]] = qualin_node.input[0]
            clip_min, clip_max = clip_range[q_output]
            clip_min_init = numpy_helper.from_array(
                np.array(clip_min, dtype=np.float32), deqlin_node.name + "_clip_min"
            )
            clip_max_init = numpy_helper.from_array(
                np.array(clip_max, dtype=np.float32), deqlin_node.name + "_clip_max"
            )
            model.graph.initializer.extend([clip_min_init, clip_max_init])
            clip_node = helper.make_node(
                "Clip",
                inputs=[
                    qualin_node.input[0],
                    deqlin_node.name + "_clip_min",
                    deqlin_node.name + "_clip_max",
                ],  # data, axes
                outputs=[deqlin_node.output[0]],
                name=deqlin_node.name + "_clip",
            )
            clip_nodes_to_add.append(clip_node)
            cnt += 1

    for clip_node in clip_nodes_to_add:
        graph.node.append(clip_node)
    logger.info("Replaced %d QuantizeLinear and DequantizeLinear pairs with Clip", cnt)


def transform_remove_qdq(model, keep_clip_after_inputs=False):
    q_output_to_q_node_map, dq_input_to_dq_node_map = {}, {}  # q_output_name -> q_node, dq_input_name -> dq_node
    graph = model.graph
    node_datatype_map = {}  # node_name -> datatype
    # Collect all the candidate Q and DQ nodes
    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # check no output or if multiple connected nodes
            if not node.output[0] or len(node.output) > 1:
                continue
            q_output_to_q_node_map[node.output[0]] = node
            node_datatype_map[node.output[0]] = get_initializer_by_name(model, node.input[2]).dtype
        elif node.op_type == "DequantizeLinear":
            if not node.input[0]:
                continue
            dq_input_to_dq_node_map[node.input[0]] = node

    qdq_node_pair_output_to_input_map = {}  # qd output -> q input
    qdq_node_pair_input_to_output_map = {}  # q input -> qd output
    clip_nodes_to_add = []
    cnt = 0

    # Find the Q and DQ node pairs and remove them.
    # There are following scenarios:
    # 1) Node --> Q --> DQ --> Node'
    # 2) graph input --> Q --> DQ --> Node
    # 3) Node --> Q --> DQ --> graph output
    # 4) Node --> Q --> DQ ... --> Q --> DQ --> Node'
    for q_output, q_node in q_output_to_q_node_map.items():
        if q_output in dq_input_to_dq_node_map and (node_datatype_map[q_output] in ["uint16", "int16"]):
            dq_node = dq_input_to_dq_node_map[q_output]
            if keep_clip_after_inputs and q_node.input[0] in [graph_input.name for graph_input in model.graph.input]:
                # Calculate clip range if we want to keep the clip after inputs
                clip_min, clip_max = calculate_clip_range(dq_node, model)
                clip_min_init = numpy_helper.from_array(
                    np.array(clip_min, dtype=np.float32), dq_node.name + "_clip_min"
                )
                clip_max_init = numpy_helper.from_array(
                    np.array(clip_max, dtype=np.float32), dq_node.name + "_clip_max"
                )
                model.graph.initializer.extend([clip_min_init, clip_max_init])
                clip_node = helper.make_node(
                    "Clip",
                    inputs=[q_node.input[0], dq_node.name + "_clip_min", dq_node.name + "_clip_max"],  # data, axes
                    outputs=[dq_node.output[0]],
                    name=dq_node.name + "_clip",
                )
                clip_nodes_to_add.append(clip_node)
            else:
                qdq_node_pair_output_to_input_map[dq_node.output[0]] = q_node.input[0]
                qdq_node_pair_input_to_output_map[q_node.input[0]] = dq_node.output[0]

            graph.node.remove(q_node)
            graph.node.remove(dq_node)
            cnt += 1

    for clip_node in clip_nodes_to_add:
        graph.node.append(clip_node)

    # Make sure the predecessor and successor to the Q and DQ node pair are connected.
    # e.g. Node --> Q --> DQ --> Node'  =>  Node --> Node'
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            # This node's ith input is from a DQ node.
            #
            # Please note that the following while loop is for handling the #4 case mentioned above where
            # there could be multiple (ususally no more than 2) consecutive Q and DQ node pairs which are connected
            # e.g. Node --> Q --> DQ --> Q --> DQ --> Node'
            if input_name in qdq_node_pair_output_to_input_map:
                qdq_node_pair_output = input_name
                num_connected_qdq_node_pair = 0
                while True:
                    qdq_node_pair_input = qdq_node_pair_output_to_input_map[qdq_node_pair_output]
                    if qdq_node_pair_input not in qdq_node_pair_output_to_input_map:
                        node.input[i] = qdq_node_pair_input
                        break
                    qdq_node_pair_output = qdq_node_pair_input

                    num_connected_qdq_node_pair += 1
                    if num_connected_qdq_node_pair > 5:
                        logger.info(
                            "Number of connected QDQ node pair is %d which is not normal.",
                            num_connected_qdq_node_pair
                        )
                        sys.exit(1)

        for i, output_name in enumerate(node.output):
            if output_name in qdq_node_pair_input_to_output_map:
                graph_output_candidate = qdq_node_pair_input_to_output_map[output_name]
                for output in graph.output:
                    if output.name == graph_output_candidate:
                        output.name = node.output[i]
                        break

    logger.info("Removed %d QuantizeLinear and DequantizeLinear pairs", cnt)


def transform_remove_deqlin(model):
    def dequantize_initializer(deq_initializers, node, graph):
        node_input = node.input
        x = None
        scale = None
        zero_point = None
        for init in deq_initializers:
            if init.name == node_input[0]:
                x = numpy_helper.to_array(init).astype(np.int32)
            if init.name == node_input[1]:
                scale = numpy_helper.to_array(init).astype(np.float32)
            if init.name == node_input[2]:
                zero_point = numpy_helper.to_array(init).astype(np.int32)

        #### Start of Shape mismatch handling ####
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        if len(x.shape) != 0:
            new_shape = [1] * len(x.shape)
            new_shape[axis] = -1  # Use -1 to keep the original size at this axis
            zero_point = zero_point.reshape(new_shape)
            scale = scale.reshape(new_shape)
        #### End of Shape mismatch handling ####
        return ((x - zero_point) * scale).astype(
            np.float32
        )  # Might have type issues: X, zero_point uint16, scale float32

    cnt = 0
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    deqlin_output_initializer_mapping = {}
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and len(node.input) > 0 and node.input[0] in initializer_names:
            deq_initializers = [init for init in graph.initializer if init.name in node.input]
            dequantized_arr = dequantize_initializer(deq_initializers, node, graph).astype(np.float32)
            # Create new initializer with fresh name and float32 datatype
            new_initializer_name = node.input[0] + "_dequantized"
            dequantized_init = numpy_helper.from_array(dequantized_arr, name=new_initializer_name)
            # Remove old initializers and add new one
            # for init in deq_initializers:
            #     graph.initializer.remove(init)
            graph.initializer.append(dequantized_init)
            deqlin_output_initializer_mapping[node.output[0]] = new_initializer_name
            nodes_to_remove.append(node)
        # Replace the node input after DequantizeLinear with initializer
        for i, input_ in enumerate(node.input):
            if input_ in deqlin_output_initializer_mapping:
                node.input[i] = deqlin_output_initializer_mapping[input_]

    # Remove the nodes
    for node in nodes_to_remove:
        graph.node.remove(node)
        cnt += 1
    logger.info("Removed %d DequantizeLinear nodes", cnt)


"""
Add Unsqueeze node to model inputs if input is 2D or 3D. Special case is if there's already an Unsqueeze node
"""


def transform_non4d_model_inputs(model):
    cnt = 0
    graph = model.graph
    unsqueeze_to_add = []
    value_info_to_add = []  # Track new value_info entries to add

    for graph_input in graph.input:
        dims = len(graph_input.type.tensor_type.shape.dim)
        if dims == 2 or dims == 3:
            for node in graph.node:
                if len(node.input) > 0 and node.input[0] == graph_input.name:
                    if node.op_type == "Unsqueeze":
                        existing_unsqueeze_axes = None
                        for init in graph.initializer:
                            if init.name == node.input[1]:
                                existing_unsqueeze_axes = numpy_helper.to_array(init)
                                break
                        if existing_unsqueeze_axes is not None:
                            if len(existing_unsqueeze_axes) + dims == 4:
                                continue
                            if existing_unsqueeze_axes.tolist()[0] == -1:
                                unsqueeze_arr = np.array([-1, 0], dtype=np.int64)
                            else:
                                unsqueeze_arr = np.array(
                                    [existing_unsqueeze_axes.tolist()[0] + 1, 0], dtype=np.int64
                                )
                            new_unsqueeze_axes_name = node.name + "unsqueeze_axes_transformed"
                            unsqueeze_axes = numpy_helper.from_array(unsqueeze_arr, new_unsqueeze_axes_name)
                            graph.initializer.append(unsqueeze_axes)
                            node.input[1] = new_unsqueeze_axes_name
                    else:
                        unsqueeze_arr = (
                            np.array([0, 1], dtype=np.int64) if dims == 2 else np.array([0], dtype=np.int64)
                        )  #### changed from 0,-1 to 0,1 , if things break, change it back and 0 to 1
                        new_unsqueeze_axes_name = graph_input.name + "_unsqueeze_axes"
                        unsqueezed_input_name = graph_input.name + "_unsqueeze_input"
                        unsqueeze_axes = numpy_helper.from_array(unsqueeze_arr, new_unsqueeze_axes_name)
                        graph.initializer.append(unsqueeze_axes)
                        unsqueeze_node = helper.make_node(
                            "Unsqueeze",
                            inputs=[graph_input.name, new_unsqueeze_axes_name],
                            outputs=[unsqueezed_input_name],
                            name=graph_input.name + "_unsqueeze",
                        )
                        unsqueeze_to_add.append(unsqueeze_node)

                        # Add value_info for the new unsqueezed tensor
                        output_shape = []
                        input_shape = []

                        # Convert ONNX dimensions to proper shape list
                        for dim in graph_input.type.tensor_type.shape.dim:
                            if dim.HasField("dim_value"):
                                input_shape.append(dim.dim_value)
                            elif dim.HasField("dim_param"):
                                input_shape.append(dim.dim_param)
                            else:
                                input_shape.append(1)  # Default for unknown dimensions

                        # Calculate the new shape after unsqueeze
                        if dims == 2:
                            # [0,1] means add dimensions at positions 0 and 1
                            output_shape = [1, 1, *input_shape]
                        else:  # dims == 3
                            # [0] means add dimension at position 0
                            output_shape = [1, *input_shape]

                        # Create value_info for the unsqueezed output
                        output_vi = helper.make_tensor_value_info(
                            unsqueezed_input_name, graph_input.type.tensor_type.elem_type, output_shape
                        )
                        value_info_to_add.append(output_vi)

                        for graph_node in graph.node:
                            for i, node_input_name in enumerate(graph_node.input):
                                if node_input_name == graph_input.name:
                                    graph_node.input[i] = unsqueezed_input_name
                    cnt += 1

    # Add all new unsqueeze nodes
    for node in unsqueeze_to_add:
        graph.node.append(node)

    # Add all new value_info entries
    for vi in value_info_to_add:
        graph.value_info.append(vi)

    logger.info("Added/Updated %d Unsqueeze nodes and %d value_info entries", cnt, len(value_info_to_add))


# Add Squeeze to non 4D model outputs
def transform_non4d_model_outputs(model):
    def is_squeeze_clip_output_pattern(squeeze_node, model):
        squeeze_output = squeeze_node.output[0]
        graph_output_names = [output.name for output in model.graph.output]
        for node in graph.node:
            if node.op_type == "Clip" and node.input[0] == squeeze_output and node.output[0] in graph_output_names:
                return True
        return False

    cnt = 0
    graph = model.graph
    for graph_output in graph.output:
        update_existing_squeeze = False
        output_dim = len(graph_output.type.tensor_type.shape.dim)
        if output_dim == 1:
            can_add_squeeze = False
            for node in graph.node:
                for node_output_name in node.output:
                    if node_output_name == graph_output.name:
                        if get_shape_from_graph(graph, node_output_name) != get_shape_from_graph(
                            graph, graph_output.name
                        ):
                            can_add_squeeze = True
            if can_add_squeeze:
                unsqueeze_input_name = graph_output.name + "_Squeeze_input"
                unsqueeze_axes_name = graph_output.name + "_Squeeze_axes"
                unsqueeze_axes_arr = np.array([0, 1, 2], dtype=np.int64)
                unsqueeze_axes = numpy_helper.from_array(unsqueeze_axes_arr, unsqueeze_axes_name)
                graph.initializer.append(unsqueeze_axes)
                unsqueeze_node = helper.make_node(
                    "Squeeze",
                    inputs=[unsqueeze_input_name, unsqueeze_axes_name],
                    outputs=[graph_output.name],
                    name=graph_output.name + "_Squeeze",
                )
                graph.node.append(unsqueeze_node)
                cnt += 1
            # Change output of previous node to unsqueeze_input_name
            for node in graph.node:
                for i, node_output_name in enumerate(node.output):
                    if node_output_name == graph_output.name:
                        if get_shape_from_graph(graph, node_output_name) != get_shape_from_graph(
                            graph, graph_output.name
                        ):
                            node.output[i] = unsqueeze_input_name
                # Handle intermediate nodes that have graph_output as input
                for i, node_input_name in enumerate(node.input):
                    if node_input_name == graph_output.name:
                        if get_shape_from_graph(graph, node_input_name) != get_shape_from_graph(
                            graph, graph_output.name
                        ):
                            node.input[i] = unsqueeze_input_name
            continue
        if output_dim < 4 and output_dim > 1:
            for node in graph.node:
                if node.op_type == "Squeeze" and (
                    node.output[0] == graph_output.name or is_squeeze_clip_output_pattern(node, model)
                ):
                    update_existing_squeeze = True
                    squeeze_axes_name = graph_output.name + "_squeeze_axes"
                    if output_dim == 2:
                        squeeze_axes_arr = np.array([0, 1], dtype=np.int64)  #### it was 3 here before, made it 1
                    if output_dim == 3:  #### this case was missing
                        squeeze_axes_arr = np.array([0, 1], dtype=np.int64)
                    squeeze_axes = numpy_helper.from_array(squeeze_axes_arr, squeeze_axes_name)
                    node.input[1] = squeeze_axes_name
                    graph.initializer.append(squeeze_axes)
            if not update_existing_squeeze:
                squeeze_input_name = graph_output.name + "_squeeze_input"
                squeeze_axes_name = graph_output.name + "_squeeze_axes"
                if output_dim == 2:
                    squeeze_axes_arr = np.array([0, 1], dtype=np.int64)
                elif output_dim == 3:
                    squeeze_axes_arr = np.array([0], dtype=np.int64)
                squeeze_axes = numpy_helper.from_array(squeeze_axes_arr, squeeze_axes_name)
                graph.initializer.append(squeeze_axes)
                squeeze_node = helper.make_node(
                    "Squeeze",
                    inputs=[squeeze_input_name, squeeze_axes_name],
                    outputs=[graph_output.name],
                    name=graph_output.name + "_squeeze",
                )
                # Change output of previous node to squeeze_input_name
                for node in graph.node:
                    for i, node_output_name in enumerate(node.output):
                        if node_output_name == graph_output.name:
                            node.output[i] = squeeze_input_name
                    # Handle intermediate nodes that have graph_output as input
                    for i, node_input_name in enumerate(node.input):
                        if node_input_name == graph_output.name:
                            node.input[i] = squeeze_input_name
                graph.node.append(squeeze_node)
                cnt += 1
    logger.info("Added %d Squeeze/Unsqueeze nodes for graph output", cnt)


def get_shape_from_graph(graph, name):
    # Check value_info
    for value_info in graph.value_info:
        if value_info.name == name:
            tensor_type = value_info.type.tensor_type
            return [d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim]
    # Check graph inputs
    for graph_input in graph.input:
        if graph_input.name == name:
            tensor_type = graph_input.type.tensor_type
            return [d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim]
    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def transform_reducemin_keepdims_GT(model):
    graph = model.graph
    for node in graph.node:
        if node.op_type == "ReduceMin" and "transformed" not in node.name:
            for attr in node.attribute:
                if attr.name == "keepdims":
                    attr.i = 1


def transform_standalone_reducesum_reducemean(
    model,
):  #### To make it suitable for running F2, i have added reducemean and to make it suitable for GT, i have added reducemin as the core tranformation is exactly same
    graph = model.graph
    reshape_counter = 0  # Add counter for unique reshape shape names
    for node in graph.node:
        # A single ReduceSum, not transformed with reshape_reducesum_to_slice_reducesum_concat
        if (
            node.op_type == "ReduceSum" or node.op_type == "ReduceMean" or node.op_type == "ReduceMax"
        ) and "transformed" not in node.name:
            # logger.info(node.name)
            # logger.info(node)
            # set keepdims = 1
            keepdims_was_zero = False
            for attr in node.attribute:
                if attr.name == "keepdims":
                    if attr.i == 0:
                        keepdims_was_zero = True
                    attr.i = 1
            if len(node.input) > 1:
                axes_initializer = node.input[1]  # axes
            else:
                axes_initializer = node.attribute[0].g.name
            initialize_to_remove = None
            new_reducesum_axes = None
            for initializer in graph.initializer:
                if initializer.name == axes_initializer:
                    current_axes = numpy_helper.to_array(initializer)

                    # Get input shape to determine if 3D or 2D
                    # Get input shape from the initializer's dims
                    reducesum_axes_name = node.name + "_axes"
                    # Get input shape from the first input's initializer
                    input_shape = get_shape_from_graph(graph, node.input[0])
                    if input_shape is not None:
                        dims = len(input_shape)
                        # Increment axes based on input dimensions
                        new_axes = current_axes
                        if current_axes[0] != -1:
                            if dims == 3:
                                new_axes = current_axes + 1
                            elif dims == 2:
                                new_axes = current_axes + 2
                            else:
                                new_axes = current_axes
                        new_reducesum_axes = numpy_helper.from_array(new_axes, reducesum_axes_name)
                        initialize_to_remove = initializer
                        node.input[1] = reducesum_axes_name

                        # If keepdims was 0, add a reshape node after the reduce operation
                        if keepdims_was_zero:
                            # Create a new reshape node
                            reshape_shape_name = f"reshape_shape_{reshape_counter}"  # Make name unique with counter
                            reshape_counter += 1  # Increment counter
                            reshape_node = onnx.helper.make_node(
                                "Reshape",
                                inputs=[node.output[0], reshape_shape_name],
                                outputs=[node.output[0] + "_reshaped"],
                                name=node.name + "_reshape",
                            )
                            # Update the original node's output to feed into reshape
                            original_output = node.output[0]
                            node.output[0] = original_output + "_temp"
                            reshape_node.input[0] = node.output[0]
                            reshape_node.output[0] = original_output

                            # Add the reshape node after the reduce node
                            graph.node.insert(list(graph.node).index(node) + 1, reshape_node)

                            if dims == 3:
                                input_shape = [1, *input_shape]
                            elif dims == 2:
                                input_shape = [1, 1, *input_shape]
                            input_shape = list(np.delete(np.array(input_shape), new_axes))
                            if len(input_shape) == 3:
                                input_shape = [1, *input_shape]
                            elif len(input_shape) == 2:
                                input_shape = [1, 1, *input_shape]
                            input_shape = list(np.array(input_shape, dtype=np.int64))

                            reshape_shape = np.array(input_shape, dtype=np.int64)  # Example shape
                            reshape_shape_initializer = numpy_helper.from_array(reshape_shape, reshape_shape_name)
                            graph.initializer.append(reshape_shape_initializer)
                    elif len(node.input) == 1:
                        for attr in node.attribute:
                            if attr.name == "axes" and attr.i != -1:
                                if len(input_shape) == 2:
                                    attr.i += 2
                                elif len(input_shape) == 3:
                                    attr.i += 1
                                break

            if initialize_to_remove is not None and new_reducesum_axes in graph.initializer:
                graph.initializer.remove(new_reducesum_axes)
            if new_reducesum_axes is not None:
                graph.initializer.append(new_reducesum_axes)


# Change Gather indices from scalar to vector, may need to update axis
def transform_gather(model):
    initializer_to_remove = None
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Gather":
            cnt += 1
            indices_initializer_name = node.input[1]

            # Check if indices are from an initializer
            existing_indices = None
            initializer_to_remove = None
            is_initializer = False

            for initializer in model.graph.initializer:
                if initializer.name == indices_initializer_name:
                    existing_indices = numpy_helper.to_array(initializer)
                    initializer_to_remove = initializer
                    is_initializer = True
                    break

            # If indices are from an initializer, transform them
            if is_initializer:
                if existing_indices is not None:
                    if existing_indices.ndim == 0:
                        indices_array = np.array([existing_indices.item()], dtype=existing_indices.dtype)
                    else:
                        indices_array = existing_indices
                else:
                    indices_array = np.array([0], dtype=np.int64)

                indices_initializer = numpy_helper.from_array(indices_array, name=indices_initializer_name)

                # Remove the old initializer if it exists
                if initializer_to_remove is not None:
                    model.graph.initializer.remove(initializer_to_remove)

                # Add the new initializer
                model.graph.initializer.append(indices_initializer)
            else:
                for node2 in model.graph.node:
                    if node2.output[0] == node.input[1]:
                        # Create unique name for the squeeze initializer
                        squeeze_initializer_name = node.name + "_squeeze_axes"
                        squeeze_initializer = numpy_helper.from_array(
                            np.array([0, 1, 2], dtype=np.int64), name=squeeze_initializer_name
                        )

                        # Create intermediate tensor name
                        squeezed_tensor_name = node2.output[0] + "_squeezed"

                        # Create squeeze node with proper inputs and attributes
                        squeeze_node = onnx.helper.make_node(
                            "Squeeze",
                            inputs=[node2.output[0], squeeze_initializer_name],
                            outputs=[squeezed_tensor_name],
                            name=node.name + "_squeeze",
                        )

                        # Get the type from the original tensor
                        original_tensor_type = None
                        for value_info in model.graph.value_info:
                            if value_info.name == node2.output[0]:
                                original_tensor_type = value_info.type.tensor_type.elem_type
                                break

                        if original_tensor_type is None:
                            # If not found in value_info, try to get from output
                            for output in model.graph.output:
                                if output.name == node2.output[0]:
                                    original_tensor_type = output.type.tensor_type.elem_type
                                    break

                        if original_tensor_type is None:
                            # Default to FLOAT if type cannot be determined
                            original_tensor_type = onnx.TensorProto.FLOAT

                        # Create new tensor value info for squeezed output
                        squeezed_output = onnx.helper.make_tensor_value_info(
                            squeezed_tensor_name,
                            original_tensor_type,
                            None,  # Shape will be inferred
                        )

                        # Update the current node's input to use squeezed output
                        node.input[1] = squeezed_tensor_name

                        # Add the new nodes and tensors to the graph
                        model.graph.node.append(squeeze_node)
                        model.graph.initializer.append(squeeze_initializer)
                        model.graph.value_info.append(squeezed_output)
                        break
            # Update axis attribute if needed (for both initializer and dynamic input cases)
            for attr in node.attribute:
                if attr.name == "axis" and attr.i == 1:
                    attr.i = 2
                    logger.info("Updated Gather node %s axis from 1 to 2", node.name)
            if not (is_initializer):
                has_axis = False
                for attr in node.attribute:
                    if attr.name == "axis":
                        has_axis = True
                        break

                if not has_axis:
                    axis_attr = onnx.helper.make_attribute("axis", 2)  # Default axis is 2
                    node.attribute.append(axis_attr)

    logger.info("Updated %d Gather nodes", cnt)


# """ Change Gather indices from scalar to vector, may need to update axis
# -
# """
def transform_gatherelements(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "GatherElements":
            cnt += 1
            indices_initializer_name = node.input[1]
            existing_indices = None
            indices_array = None
            for initializer in model.graph.initializer:
                if initializer.name == indices_initializer_name:
                    existing_indices = numpy_helper.to_array(initializer)
                    initializer_to_remove = initializer
                    break
            if existing_indices is not None:
                if existing_indices.ndim == 0:
                    indices_array = np.array([existing_indices.item()], dtype=existing_indices.dtype)
                elif existing_indices.ndim == 3:
                    # Handle 3D indices, reshape to 4D by adding dimension at front
                    logger.info("Reshaping 3D indices %s from %s to 4D", initializer.name, existing_indices.shape)
                    indices_array = np.expand_dims(existing_indices, axis=0)  # Eg.[12,64,64]->[1,12,64,64]
                else:
                    indices_array = existing_indices
            else:
                return
            indices_initializer = numpy_helper.from_array(indices_array, name=indices_initializer_name)
            # Remove the old initializer if it exists
            if initializer_to_remove is not None:
                model.graph.initializer.remove(initializer_to_remove)

            # Add the new initializer
            model.graph.initializer.append(indices_initializer)
    logger.info("Updated %d GatherElements indices", cnt)


def transform_non4d_initializers(model):
    # Purpose of need_to_expand_4D_init_names is to avoid expanding some 1D initializers such as axes, slice_begin
    need_to_expand_4d_init_names = []
    skip_init_names = []
    unary_dim_at_front_init_names = []
    avoid_expanding_4d_init_names = []
    for node in model.graph.node:
        if node.op_type in ["Div", "Sub", "Mul", "Add"]:
            need_to_expand_4d_init_names.extend(node.input)
        if node.op_type in ["MatMul"]:
            skip_init_names.extend(node.input)
        if node.op_type in ["Gemm", "Where", "Gather", "Split"]:
            unary_dim_at_front_init_names.extend(node.input)
        if node.op_type in ["Unsqueeze"]:  #### Avoid expanding 4D for Unsqueeze inputs
            avoid_expanding_4d_init_names.extend(node.input)

    initializers_to_add = []
    initializer_to_remove = []
    for initializer in model.graph.initializer:
        if (
            len(initializer.dims) == 1
            and initializer.name in need_to_expand_4d_init_names
            and initializer.name not in avoid_expanding_4d_init_names
        ):
            # 1D: [K] -> [1x1x1xK]
            initializer.dims.insert(0, 1)
            initializer.dims.insert(0, 1)
            initializer.dims.insert(0, 1)
        elif (
            len(initializer.dims) == 2
            and initializer.name in need_to_expand_4d_init_names
            and initializer.name not in avoid_expanding_4d_init_names
        ):
            initializer.dims.insert(0, 1)
            initializer.dims.insert(0, 1)
        elif (
            len(initializer.dims) == 2
            and initializer.name in unary_dim_at_front_init_names
            and initializer.name not in avoid_expanding_4d_init_names
        ):
            # 2D: [K, C] -> [1, 1, K, C]
            new_dims = [1, 1, *list(initializer.dims)]
            initializer.dims[:] = new_dims
        elif (
            len(initializer.dims) == 2
            and initializer.name not in skip_init_names
            and initializer.name not in avoid_expanding_4d_init_names
        ):
            # 2D [CxK] -> [KxCx1x1]
            c, k = initializer.dims[0], initializer.dims[1]
            init_arr = numpy_helper.to_array(initializer)
            transposed_arr = init_arr.T
            reshaped_arr = np.reshape(transposed_arr, (k, c, 1, 1))
            new_initializer = numpy_helper.from_array(reshaped_arr, initializer.name)
            initializer_to_remove.append(initializer)
            initializers_to_add.append(new_initializer)
            initializer.dims[0], initializer.dims[1] = k, c
            initializer.dims.insert(2, 1)
            initializer.dims.insert(3, 1)

            # new_dims = [1] + list(initializer.dims)
        elif len(initializer.dims) == 3 and initializer.name not in avoid_expanding_4d_init_names:
            # initializer.dims.insert(0, 1)
            new_dims = [1, *list(initializer.dims)]
            initializer.dims[:] = new_dims

    [model.graph.initializer.remove(init) for init in initializer_to_remove]
    [model.graph.initializer.append(init) for init in initializers_to_add]


def transform_remove_all_tensor_value_shapes(model):
    for value_info in model.graph.value_info:
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField("shape"):
            tensor_type.ClearField("shape")


def transform_non4d_reshape(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Reshape":
            # Check if axes
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    if init.dims[0] == 2:
                        # eg [-1, 512] -> [1, 1, -1, 512]
                        old_shape = numpy_helper.to_array(init)
                        init.dims[0] = 4
                        # add 1s at the beginning
                        new_shape = np.concatenate([np.array([1, 1]), old_shape]).astype(np.int64)
                        new_init = numpy_helper.from_array(new_shape, name=init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        cnt += 1

                    if init.dims[0] == 3:
                        # Get the input shape for this Reshape node
                        input_name = node.input[0]
                        input_shape = None
                        # Try to get input shape from value_info, initializer, or graph input
                        tensor_name_dim_map = get_tensor_shape_map(model.graph.value_info)
                        initializer_dim_map = {init.name: list(init.dims) for init in model.graph.initializer}
                        if input_name in tensor_name_dim_map:
                            input_shape = tensor_name_dim_map[input_name]
                        elif input_name in initializer_dim_map:
                            input_shape = initializer_dim_map[input_name]
                        else:
                            for graph_input in model.graph.input:
                                if graph_input.name == input_name:
                                    tensor_type = graph_input.type.tensor_type
                                    input_shape = []
                                    if tensor_type.HasField("shape"):
                                        for d in tensor_type.shape.dim:
                                            if d.HasField("dim_value"):
                                                input_shape.append(d.dim_value)
                                            else:
                                                input_shape.append(0)
                                    break

                        old_shape = numpy_helper.to_array(init)
                        # Insert 1 at the beginning to make it 4D
                        new_shape = np.insert(old_shape, 0, 1).astype(np.int64)
                        # Replace 0s with input_shape[axis-1]
                        for i in range(1, 4):  # Only axes 1,2,3 (since 0 is the new leading 1)
                            if new_shape[i] == 0 and input_shape is not None and (i - 1) < len(input_shape):
                                new_shape[i] = input_shape[i - 1]
                        new_init = numpy_helper.from_array(new_shape, name=init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        cnt += 1
                    if init.dims[0] == 5:
                        old_shape = numpy_helper.to_array(init)
                        if old_shape[0] == 1:
                            init.dims[0] = 4
                            new_shape = old_shape[1:].astype(np.int64)
                            new_init = numpy_helper.from_array(new_shape, name=init.name)
                            model.graph.initializer.remove(init)
                            model.graph.initializer.append(new_init)
                            cnt += 1
    logger.info("Updated %d non4D Reshape nodes", cnt)


def transform_non4d_expand(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Expand":
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    if init.dims[0] == 3:
                        old_shape = numpy_helper.to_array(init)
                        init.dims[0] = 4
                        new_shape = np.insert(old_shape, 0, 1).astype(np.int64)
                        new_init = numpy_helper.from_array(new_shape, name=init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        cnt += 1
    logger.info("Updated %d non4D Expand nodes", cnt)


def transform_non4d_tile(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Tile":
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    if init.dims[0] == 3:
                        old_shape = numpy_helper.to_array(init)
                        init.dims[0] = 4
                        new_shape = np.insert(old_shape, 0, 1).astype(np.int64)
                        new_init = numpy_helper.from_array(new_shape, name=init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        cnt += 1
    logger.info("Updated %d non4D Tile nodes", cnt)


"""
perm attribute of Transpose
- 2D: [T0, T1] -> [1, 1, T0 + 2, T1 + 2]
- 3D: [T0, T1, T2] -> [1, T0 + 1. T1 + 1, T2 + 1]
"""


def transform_non4d_transpose(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm" and len(attr.ints) == 2:
                    old_perm = list(attr.ints)
                    new_perm = [0, 1, old_perm[0] + 2, old_perm[1] + 2]
                    attr.ints[:] = new_perm
                    cnt += 1
                elif attr.name == "perm" and len(attr.ints) == 3:
                    # [0, 2, 1] -> [0, 3, 2, 1]
                    old_perm = list(attr.ints)
                    new_perm = [0, old_perm[0] + 1, old_perm[1] + 1, old_perm[2] + 1]
                    attr.ints[:] = new_perm
                    cnt += 1
                elif attr.name == "perm" and len(attr.ints) == 5:
                    old_perm = list(attr.ints)
                    new_perm = [old_perm[0] - 1, old_perm[1] - 1, old_perm[2] - 1, old_perm[3] - 1]
                    for i in range(len(new_perm)):
                        if new_perm[i] == -1:
                            new_perm[i] = len(old_perm) - 2
                    attr.ints[:] = new_perm
                    cnt += 1
    logger.info("Updated %d non4D Transpose nodes", cnt)


# # Transform Slice axes of non4D tensors
# def transform_non4d_slice(model):
#     cnt = 0
#     for node in model.graph.node:
#         # Skip transformed nodes
#         if node.op_type == 'Slice' and not 'transformed_' in node.input[3]:
#             new_init_to_add = None
#             for init in model.graph.initializer:
#                 if init.name == node.input[3] and init.dims[0] == 1:
#                     new_init_name = node.name + '_axes'
#                     new_init_to_add = numpy_helper.from_array(np.array([-1], dtype=np.int64), name=new_init_name)
#                     node.input[3] = new_init_name
#                     cnt += 1
#             if new_init_to_add is not None:
#                 model.graph.initializer.append(new_init_to_add)
#     logger.info(f"Updated {cnt} non4D Slice axes")


# Transform LpNormalization axes of non4D tensors
def transform_non4d_lpnorm(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == "LpNormalization":
            for attr in node.attribute:
                if attr.name == "axis":
                    attr.i = -1
            cnt += 1
    logger.info("Updated %d non4D LpNormalization axes", cnt)


# Flatten to Reshape
def transform_flatten(model):
    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == "Flatten":
            reshape_axes = numpy_helper.from_array(
                np.array([1, 1, 1, -1], dtype=np.int64), name=node.name + "_reshape_axes"
            )
            reshape_node = helper.make_node(
                "Reshape", inputs=[node.input[0], reshape_axes.name], outputs=node.output, name=node.name + "_reshape"
            )
            nodes_to_remove.append(node)
            model.graph.initializer.append(reshape_axes)
            model.graph.node.append(reshape_node)
    # Remove flatten node(s)
    for node in nodes_to_remove:
        model.graph.node.remove(node)


# Debug function to add intermediate tensors to outputs
def transform_add_intermediate_tensors_to_outputs(model, intermediate_tensor_to_add=None):
    # Get existing output names
    existing_outputs = {output.name for output in model.graph.output}

    # Collect all intermediate tensor names from node outputs
    if intermediate_tensor_to_add is None:
        for node in model.graph.node:
            for output in node.output:
                if output and output not in existing_outputs:
                    intermediate_tensor_to_add.add(output)

    # Create ValueInfoProto for each intermediate tensor
    for tensor_name in intermediate_tensor_to_add:
        # Create a new output with default type FLOAT
        output_info = onnx.helper.make_tensor_value_info(
            tensor_name,
            onnx.TensorProto.FLOAT,  # Default to FLOAT type
            None,  # Shape will be inferred if possible
        )
        model.graph.output.append(output_info)


def transform_remove_unused_initializers(model):
    """Remove initializers that are not used as inputs to any node in the graph.

    Args:
        model: An ONNX ModelProto object

    Returns:
        The modified model with unused initializers removed

    """
    graph = model.graph

    # Collect all node inputs
    used_inputs = set()
    for node in graph.node:
        used_inputs.update(node.input)

    # Also consider graph outputs as used
    for output in graph.output:
        used_inputs.add(output.name)

    # Find initializers that are used
    used_initializers = []
    removed_count = 0

    for initializer in graph.initializer:
        if initializer.name in used_inputs:
            used_initializers.append(initializer)
        else:
            removed_count += 1

    # Clear and reset initializers
    graph.ClearField("initializer")
    graph.initializer.extend(used_initializers)

    logger.info("Removed %d unused initializers", removed_count)


###
# Onnxscript transform
###

"""
FROM
    x, shape
    |
Reshape axes
    |   /
ReduceSum
    |
reducesum_output
TO
      x
   /        \
Slice      Slice
  |           |
ReduceSum   ReduceSum
    \\       /
    Concat
      |
reducesum_output
"""


def reshape_reducesum_pattern(op, x, shape, axes):
    reshape_output = op.Reshape(x, shape)
    reducesum_output = op.ReduceSum(reshape_output, axes)
    return reducesum_output


def slice_reducesum_concat(op, x, shape, axes):
    slice_0_starts = op.initializer(ir.tensor([0], dtype=ir.DataType.INT64, name=x.name + "_slice_0_starts"))
    slice_0_ends = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_0_ends"))
    slice_1_starts = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_1_starts"))
    slice_1_ends = op.initializer(ir.tensor([8], dtype=ir.DataType.INT64, name=x.name + "_slice_1_ends"))
    slice_reduce_axes = op.initializer(ir.tensor([3], dtype=ir.DataType.INT64, name=x.name + "_transformed_axes"))
    slice_output0 = op.Slice(x, slice_0_starts, slice_0_ends, slice_reduce_axes)
    slice_output1 = op.Slice(x, slice_1_starts, slice_1_ends, slice_reduce_axes)
    reducesum_output0 = op.ReduceSum(slice_output0, slice_reduce_axes)
    reducesum_output1 = op.ReduceSum(slice_output1, slice_reduce_axes)
    return op.Concat(reducesum_output0, reducesum_output1, axis=3)


def transform_reshape_reducesum(model):
    reshape_reducesum_rule = pattern.RewriteRule(reshape_reducesum_pattern, slice_reducesum_concat, verbose=10)
    model = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[reshape_reducesum_rule],
    )
    return model


"""
FROM
    (x)
    |
  Reshape
    |
   Clip
    |
ReduceSum
    |
(reducesum_output)
TO
      (x)
   /        \
Slice      Slice
  |           |
Clip        Clip
  |           |
ReduceSum   ReduceSum
    \\       /
    Concat
      |
(reducesum_output)
"""


def transform_reshape_clip_reducesum(model):
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    counter = 0
    for node in list(graph.node):
        if node.op_type == "Reshape":
            reshape_out = node.output[0]
            # Find Clip
            clip_nodes = [n for n in graph.node if n.input and n.input[0] == reshape_out and n.op_type == "Clip"]
            if not clip_nodes:
                continue
            clip_node = clip_nodes[0]
            clip_out = clip_node.output[0]
            # Find ReduceSum
            rs_nodes = [n for n in graph.node if n.input and n.input[0] == clip_out and n.op_type == "ReduceSum"]
            if not rs_nodes:
                continue
            rs_node = rs_nodes[0]

            # Remove old nodes
            nodes_to_remove.extend([node, clip_node, rs_node])

            # Get parameters
            input_tensor = node.input[0]  # Use the input to the original Reshape
            clip_min = clip_node.input[1] if len(clip_node.input) > 1 else None
            clip_max = clip_node.input[2] if len(clip_node.input) > 2 else None
            axes = rs_node.input[1] if len(rs_node.input) > 1 else None
            output_name = rs_node.output[0]

            # Create slice parameters (example: split along axis 3, first half and second half)
            slice_0_starts = helper.make_tensor(
                name=f"slice_0_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[0]
            )
            slice_0_ends = helper.make_tensor(
                name=f"slice_0_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[4]
            )
            slice_1_starts = helper.make_tensor(
                name=f"slice_1_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[4]
            )
            slice_1_ends = helper.make_tensor(
                name=f"slice_1_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[8]
            )
            slice_axes = helper.make_tensor(
                name=f"slice_axes_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[3]
            )

            graph.initializer.extend([slice_0_starts, slice_0_ends, slice_1_starts, slice_1_ends, slice_axes])

            # Unique names for all new nodes
            slice0_out = f"slice0_out_{counter}"
            clip0_out = f"clip0_out_{counter}"
            rs0_out = f"rs0_out_{counter}"
            slice1_out = f"slice1_out_{counter}"
            clip1_out = f"clip1_out_{counter}"
            rs1_out = f"rs1_out_{counter}"

            # Slice 0 branch
            slice0 = helper.make_node(
                "Slice",
                [input_tensor, f"slice_0_starts_{counter}", f"slice_0_ends_{counter}", f"slice_axes_{counter}"],
                [slice0_out],
            )
            clip0_inputs = [slice0_out]
            if clip_min is not None:
                clip0_inputs.append(clip_min)
            if clip_max is not None:
                clip0_inputs.append(clip_max)
            clip0 = helper.make_node("Clip", clip0_inputs, [clip0_out])
            rs0 = helper.make_node("ReduceSum", [clip0_out, axes], [rs0_out])
            # Slice 1 branch
            slice1 = helper.make_node(
                "Slice",
                [input_tensor, f"slice_1_starts_{counter}", f"slice_1_ends_{counter}", f"slice_axes_{counter}"],
                [slice1_out],
            )
            clip1_inputs = [slice1_out]
            if clip_min is not None:
                clip1_inputs.append(clip_min)
            if clip_max is not None:
                clip1_inputs.append(clip_max)
            clip1 = helper.make_node("Clip", clip1_inputs, [clip1_out])
            rs1 = helper.make_node("ReduceSum", [clip1_out, axes], [rs1_out])
            # Concat
            concat = helper.make_node("Concat", [rs0_out, rs1_out], [output_name], axis=3)

            nodes_to_add.extend([slice0, clip0, rs0, slice1, clip1, rs1, concat])
            counter += 1

    logger.info("Updated %d Reshape nodes with Clip ReduceSum", counter)
    # Remove old nodes
    for n in nodes_to_remove:
        graph.node.remove(n)
    # Add new nodes
    for n in nodes_to_add:
        graph.node.append(n)
    return model


"""
FROM
    (x)
    |
  Reshape
    |
   QuantizeLinear
    |
  DequantizeLinear
    |
ReduceSum
    |
(reducesum_output)
TO
      (x)
   /        \
Slice      Slice
  |           |
QuantizeLinear QuantizeLinear
  |           |
DequantizeLinear DequantizeLinear
  |           |
ReduceSum   ReduceSum
    \\       /
    Concat
      |
(reducesum_output)
"""


def transform_reshape_qdq_reducesum_generic(model):
    from onnx import TensorProto, helper

    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    counter = 0
    for node in list(graph.node):
        if node.op_type == "Reshape":
            reshape_out = node.output[0]
            # Find QuantizeLinear
            qdq = [n for n in graph.node if n.input and n.input[0] == reshape_out and n.op_type == "QuantizeLinear"]
            if not qdq:
                continue
            q_node = qdq[0]
            dq_out = q_node.output[0]
            # Find DequantizeLinear
            dq_nodes = [n for n in graph.node if n.input and n.input[0] == dq_out and n.op_type == "DequantizeLinear"]
            if not dq_nodes:
                continue
            dq_node = dq_nodes[0]
            reducesum_in = dq_node.output[0]
            # Find ReduceSum
            rs_nodes = [n for n in graph.node if n.input and n.input[0] == reducesum_in and n.op_type == "ReduceSum"]
            if not rs_nodes:
                continue
            rs_node = rs_nodes[0]

            # Remove old nodes
            nodes_to_remove.extend([node, q_node, dq_node, rs_node])

            # Get parameters
            input_tensor = node.input[0]  # Use the input to the original Reshape
            q_scale = q_node.input[1]
            q_zero_point = q_node.input[2]
            axes = rs_node.input[1] if len(rs_node.input) > 1 else None
            output_name = rs_node.output[0]

            # Create slice parameters (example: split along axis 3, first half and second half)
            slice_0_starts = helper.make_tensor(
                name=f"slice_0_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[0]
            )
            slice_0_ends = helper.make_tensor(
                name=f"slice_0_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[4]
            )
            slice_1_starts = helper.make_tensor(
                name=f"slice_1_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[4]
            )
            slice_1_ends = helper.make_tensor(
                name=f"slice_1_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[8]
            )
            slice_axes = helper.make_tensor(
                name=f"slice_axes_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[3]
            )

            graph.initializer.extend([slice_0_starts, slice_0_ends, slice_1_starts, slice_1_ends, slice_axes])

            # Unique names for all new nodes
            slice0_out = f"slice0_out_{counter}"
            q0_out = f"q0_out_{counter}"
            dq0_out = f"dq0_out_{counter}"
            rs0_out = f"rs0_out_{counter}"
            slice1_out = f"slice1_out_{counter}"
            q1_out = f"q1_out_{counter}"
            dq1_out = f"dq1_out_{counter}"
            rs1_out = f"rs1_out_{counter}"

            # Slice 0 branch
            slice0 = helper.make_node(
                "Slice",
                [input_tensor, f"slice_0_starts_{counter}", f"slice_0_ends_{counter}", f"slice_axes_{counter}"],
                [slice0_out],
            )
            q0 = helper.make_node("QuantizeLinear", [slice0_out, q_scale, q_zero_point], [q0_out])
            dq0 = helper.make_node("DequantizeLinear", [q0_out, q_scale, q_zero_point], [dq0_out])
            rs0 = helper.make_node("ReduceSum", [dq0_out, axes], [rs0_out])
            # Slice 1 branch
            slice1 = helper.make_node(
                "Slice",
                [input_tensor, f"slice_1_starts_{counter}", f"slice_1_ends_{counter}", f"slice_axes_{counter}"],
                [slice1_out],
            )
            q1 = helper.make_node("QuantizeLinear", [slice1_out, q_scale, q_zero_point], [q1_out])
            dq1 = helper.make_node("DequantizeLinear", [q1_out, q_scale, q_zero_point], [dq1_out])
            rs1 = helper.make_node("ReduceSum", [dq1_out, axes], [rs1_out])
            # Concat
            concat = helper.make_node("Concat", [rs0_out, rs1_out], [output_name], axis=3)

            nodes_to_add.extend([slice0, q0, dq0, rs0, slice1, q1, dq1, rs1, concat])
            counter += 1

    logger.info("Updated %d Reshape nodes with QDQ ReduceSum", counter)
    # Remove old nodes
    for n in nodes_to_remove:
        graph.node.remove(n)
    # Add new nodes
    for n in nodes_to_add:
        graph.node.append(n)
    return model


"""
FROM
data   axes   keepdims
    |   /     /
ReduceMax
    |
reducemax_output
TO
data   axes   keepdims
    |   /     /
ReduceMax
    |    reshape_shape
    |   /
Reshape
    |
reducemax_output
"""


def reducemax_pattern(op, data, axes, keepdims):
    return op.Reducemax(data, axes, keepdims)


def reducemax_reshape(op, data, axes, keepdims):
    new_axes = op.initializer(ir.tensor([3], dtype=ir.DataType.INT64, name=data.name + "_reducemax_axes"))
    new_keepdims = op.initializer(ir.value(1, dtype=ir.DataType.INT64, name=data.name + "_reducemax_keepdim"))
    reducemax_output = op.Reducemax(data, new_axes, new_keepdims)
    reshape_shape = op.initializer(
        ir.tensor([1, 1, 1, 3600], dtype=ir.DataType.INT64, name=data.name + "_reshape_shape")
    )
    return op.Reshape(reducemax_output, reshape_shape)


def transform_reducemax(model):
    reducemax_rule = pattern.RewriteRule(reducemax_pattern, reducemax_reshape, verbose=10)
    model = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[reducemax_rule],
    )
    return model


def transform_argmax(model):
    """For all ArgMax nodes, if input is 2D, increment axis by 2; if input is 3D, increment axis by 1.
    
    Also set keepdims=1.
    If keepdims=0, add a Reshape to shift the dimension to the beginning.
    """
    graph = model.graph

    def get_shape_from_graph(graph, name):
        # Check value_info
        for value_info in graph.value_info:
            if value_info.name == name:
                tensor_type = value_info.type.tensor_type
                return [d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim]
        # Check graph inputs
        for graph_input in graph.input:
            if graph_input.name == name:
                tensor_type = graph_input.type.tensor_type
                return [d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim]
        # Check initializers
        for init in graph.initializer:
            if init.name == name:
                return list(init.dims)
        return None

    cnt_2d = 0
    cnt_3d = 0
    for node in graph.node:
        axis = 0
        if node.op_type == "ArgMax":
            input_shape = get_shape_from_graph(graph, node.input[0])
            if input_shape is not None:
                dims = len(input_shape)
                keepdims = 1
                for attr in node.attribute:
                    if attr.name == "axis":
                        if dims == 2:
                            attr.i += 2
                            cnt_2d += 1
                        elif dims == 3:
                            attr.i += 1
                            cnt_3d += 1
                        axis = attr.i
                    elif attr.name == "keepdims":
                        keepdims = attr.i
                        attr.i = 1

                # If keepdims was 0, add a Reshape to shift the dimension to the beginning
                if keepdims == 0:
                    current_output = node.output[0]
                    # Create intermediate output name
                    intermediate_output = node.output[0] + "_intermediate"
                    node.output[0] = intermediate_output

                    # Create shape initializer for Reshape
                    reshape_shape_name = node.name + "_reshape_shape"
                    # For 2D input: [1, N] -> [1, 1, 1, N]
                    # For 3D input: [1, M, N] -> [1, 1, 1, N]
                    axes = input_shape
                    if len(axes) == 2:
                        axes = [1, 1, *axes]
                    elif len(axes) == 3:
                        axes = [1, *axes]
                    axes.pop(axis)
                    axes.insert(0, 1)
                    reshape_shape = numpy_helper.from_array(np.array(axes, dtype=np.int64), reshape_shape_name)
                    graph.initializer.append(reshape_shape)

                    # Create Reshape node
                    reshape_node = helper.make_node(
                        "Reshape",
                        inputs=[intermediate_output, reshape_shape_name],
                        outputs=[current_output],
                        name=node.name + "_reshape",
                    )
                    graph.node.append(reshape_node)

    logger.info(
        "Updated %d ArgMax nodes with 2D inputs (axis+2), %d with 3D inputs (axis+1), all with keepdims=1",
        cnt_2d,
        cnt_3d,
    )


###
# public helper functions
###
def count_ops(model):
    ops = defaultdict(int)
    for node in model.graph.node:
        ops[node.op_type] += 1
        if node.op_type not in ops:
            ops.add(node.op_type)
    logger.info("=== Graph ops count ===")
    for op_name, cnt in sorted(ops.items()):
        logger.info("%d %s", cnt, op_name)


def execute_shape_inference(input_model, output_model, save_as_external_data=False):
    try:
        # Construct command for symbolic shape inference
        if save_as_external_data:
            save_as_external_data_flag = "--save_as_external_data"
        else:
            save_as_external_data_flag = ""
        data_file_name = output_model.split("\\")[-1].replace(".onnx", ".data")
        external_data_file_flag = ""
        all_tensors_to_one_file_flag = ""
        if save_as_external_data:
            external_data_file_flag = "--external_data_location " + data_file_name
            all_tensors_to_one_file_flag = "--all_tensors_to_one_file"
        symbolic_shape_infer_cmd = (
            f"python TensorRT-Model-Optimizer-main\\modelopt\\onnx\\npu_transform\\symbolic_shape_infer.py "
            # f"py symbolic_shape_infer.py "
            f"--input {input_model} "
            f"--output {output_model} "
            # f"--auto_merge"
            f"--auto_merge --verbose 3 "
            f"{save_as_external_data_flag} "  # --save_as_external_data
            f"{external_data_file_flag} "
            f"{all_tensors_to_one_file_flag} "
        )
        # Run the command
        logger.info("Running: %s", symbolic_shape_infer_cmd)
        result = subprocess.run(symbolic_shape_infer_cmd, shell=True, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            logger.info("Shape inference completed successfully. Output saved to %s", output_model)
            logger.info(result.stdout)
        else:
            logger.info("Shape inference failed with error:")
            logger.info(result.stderr)
    except Exception as e:
        logger.info("Error running symbolic shape inference: %s", str(e))


def all_tensors_are_4d(model):
    return False


def transform_split(model):
    graph = model.graph
    for node in graph.node:
        if node.op_type == "Split":
            input_shape = get_shape_from_graph(graph, node.input[0])
            for attr in node.attribute:
                if attr.name == "axis" and attr.i != -1:
                    if len(input_shape) == 2:
                        attr.i += 2
                    elif len(input_shape) == 3:
                        attr.i += 1
                    elif len(input_shape) == 5:
                        attr.i -= 1
                    break
    return model


def transform_topk(model):
    graph = model.graph
    for node in graph.node:
        if node.op_type == "TopK":
            input_shape = get_shape_from_graph(graph, node.input[0])
            for attr in node.attribute:
                if attr.name == "axis" and attr.i != -1:
                    if len(input_shape) == 2:
                        attr.i += 2
                    elif len(input_shape) == 3:
                        attr.i += 1
                    break
    return model


def transform_non4d_concat_axis(model):
    """For all Concat nodes, if all inputs are 3D, increment axis by 1; if all are 2D, increment axis by 2.

    Should be applied at the beginning of the transform pipeline.
    """
    cnt_3d = 0
    cnt_2d = 0
    graph = model.graph
    # Build a map from tensor name to shape
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
    initializer_dim_map = {init.name: list(init.dims) for init in graph.initializer}

    for node in graph.node:
        if node.op_type == "Concat":
            input_shapes = []
            for inp in node.input:
                shape = tensor_name_dim_map.get(inp)
                if shape is None and inp in initializer_dim_map:
                    shape = initializer_dim_map[inp]
                if shape is not None:
                    input_shapes.append(shape)
            # Only proceed if all input shapes are found
            if len(input_shapes) == len(node.input):
                dims_set = {len(s) for s in input_shapes}
                if len(dims_set) == 1:
                    dims = dims_set.pop()
                    for attr in node.attribute:
                        if attr.name == "axis" and attr.i != -1:
                            if dims == 3:
                                attr.i += 1
                                cnt_3d += 1
                            elif dims == 2:
                                attr.i += 2
                                cnt_2d += 1
    logger.info("Updated %d Concat nodes with 3D inputs (axis+1), %d with 2D inputs (axis+2)", cnt_3d, cnt_2d)


def transform_squeeze_unsqueeze_to_reshape(model):
    """Detect and convert Squeeze-Unsqueeze pairs to a single Reshape operation.
    
    This optimization helps reduce unnecessary dimension operations in the graph.

    Pattern:
    Input -> Squeeze -> Unsqueeze -> Output
    becomes
    Input -> Reshape -> Output

    The Reshape operation will use a fixed shape of [1,1,-1,1]
    """
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []

    # Create a mapping of node output names to their consumers
    output_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)

    # Find Squeeze-Unsqueeze pairs
    for node in graph.node:
        if node.op_type == "Squeeze":
            squeeze_output = node.output[0]
            consumers = output_to_consumers.get(squeeze_output, [])

            # Check if this Squeeze is followed by an Unsqueeze
            for consumer in consumers:
                if consumer.op_type == "Unsqueeze":
                    # Create a new Reshape node
                    reshape_name = f"{node.name}_to_{consumer.name}_reshape"
                    reshape_shape_name = f"{reshape_name}_shape"

                    # Create shape initializer for Reshape with fixed shape [1,1,-1,1]
                    reshape_shape = numpy_helper.from_array(np.array([1, 1, -1, 1], dtype=np.int64), reshape_shape_name)
                    graph.initializer.append(reshape_shape)

                    # Create Reshape node
                    reshape_node = helper.make_node(
                        "Reshape",
                        inputs=[node.input[0], reshape_shape_name],
                        outputs=[consumer.output[0]],
                        name=reshape_name,
                    )

                    # Add new node and mark old nodes for removal
                    nodes_to_add.append(reshape_node)
                    nodes_to_remove.extend([node, consumer])

                    # Update consumers of the Unsqueeze output to use the Reshape output
                    for next_node in graph.node:
                        for i, input_name in enumerate(next_node.input):
                            if input_name == consumer.output[0]:
                                next_node.input[i] = reshape_node.output[0]

    # Remove old nodes and add new ones
    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)

    for node in nodes_to_add:
        graph.node.append(node)

    logger.info(
        "Converted %d Squeeze-Unsqueeze pairs to Reshape operations with shape [1,1,-1,1]", len(nodes_to_remove) // 2
    )


def get_input_shape_from_graph_inputs(graph, input_name):
    for graph_input in graph.input:
        if graph_input.name == input_name:
            tensor_type = graph_input.type.tensor_type
            shape = []
            if tensor_type.HasField("shape"):
                for d in tensor_type.shape.dim:
                    if d.HasField("dim_value"):
                        shape.append(d.dim_value)
                    else:
                        shape.append(0)
            return shape
    return None


def transform_non4d_slice_axis(model):
    """For all Slice nodes, if input is 2D or 3D, increment axis by 1.
    
    Should be applied at the beginning of the transform pipeline.
    """
    cnt_2d = 0
    cnt_3d = 0
    graph = model.graph
    # Build a map from tensor name to shape
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
    initializer_dim_map = {init.name: list(init.dims) for init in graph.initializer}

    for node in graph.node:
        if node.op_type == "Slice":
            input_shape = tensor_name_dim_map.get(node.input[0])
            if input_shape is None and node.input[0] in initializer_dim_map:
                input_shape = initializer_dim_map[node.input[0]]
            if input_shape is None:
                input_shape = get_input_shape_from_graph_inputs(graph, node.input[0])

            if input_shape is not None:
                dims = len(input_shape)
                axes_input = node.input[3]

                # Find the axes initializer
                for init in graph.initializer:
                    if init.name == axes_input:
                        axes = numpy_helper.to_array(init)

                        if axes[0] != -1 and (dims == 2 or dims == 3):
                            if dims == 2:
                                new_axes = np.array([axes[0] + 2], dtype=axes.dtype)
                            elif dims == 3:
                                new_axes = np.array([axes[0] + 1], dtype=axes.dtype)
                            else:
                                new_axes = np.array([-1], dtype=axes.dtype)
                            # If this initializer is used by more than one input, clone it

                            uses = sum([axes_input == n for n in node.input])
                            if uses > 1 or any(axes_input in n.input[1:] for n in graph.node if n != node):
                                # Create a new initializer for axes

                                new_axes_name = axes_input + "_axes_only"
                                new_axes_init = numpy_helper.from_array(new_axes, name=new_axes_name)
                                graph.initializer.append(new_axes_init)
                                node.input[3] = new_axes_name
                            else:
                                # Safe to update in place
                                new_axes_init = numpy_helper.from_array(new_axes, name=init.name)
                                graph.initializer.remove(init)
                                graph.initializer.append(new_axes_init)

                            if dims == 2:
                                cnt_2d += 1
                            else:
                                cnt_3d += 1
                        break

    logger.info("Updated %d Slice nodes with 2D inputs (axis+1), %d with 3D inputs (axis+1)", cnt_2d, cnt_3d)


def transform_fix_instancenorm_channel_mismatch_PSD6(model):
    """For each InstanceNormalization node, if the input's channel dimension does not match the scale length,

    insert a Reshape before the node that rotates the input shape left by one (e.g., [1,2,3,4] -> [2,3,4,1]).
    """
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
    initializer_dim_map = {init.name: list(init.dims) for init in graph.initializer}

    nodes_to_add = []
    for node in list(graph.node):
        if node.op_type == "InstanceNormalization":
            input_name = node.input[0]
            scale_name = node.input[1]

            # Get input shape
            input_shape = tensor_name_dim_map.get(input_name)
            if input_shape is None and input_name in initializer_dim_map:
                input_shape = initializer_dim_map[input_name]
            if input_shape is None:
                # Try graph inputs
                for graph_input in graph.input:
                    if graph_input.name == input_name:
                        tensor_type = graph_input.type.tensor_type
                        input_shape = []
                        if tensor_type.HasField("shape"):
                            for d in tensor_type.shape.dim:
                                if d.HasField("dim_value"):
                                    input_shape.append(d.dim_value)
                                else:
                                    input_shape.append(0)
                        break
            if input_shape is None or len(input_shape) < 2:
                continue  # Can't process if shape unknown or not enough dims

            # Get scale shape
            scale_shape = None
            for init in graph.initializer:
                if init.name == scale_name:
                    scale_shape = list(init.dims)
                    break
            if scale_shape is None or len(scale_shape) != 1:
                continue  # Can't process if scale shape unknown or not 1D
            if len(input_shape) == 3:
                C = input_shape[0]
                scale_len = scale_shape[0]
                if scale_len != C:
                    # Need to insert Reshape
                    new_shape = [*input_shape, 1]  # rotate left
                    reshape_shape_name = node.name + "_rotateleft_shape"
                    reshape_shape_init = numpy_helper.from_array(
                        np.array(new_shape, dtype=np.int64), reshape_shape_name
                    )
                    graph.initializer.append(reshape_shape_init)

                    reshape_output_name = node.input[0] + "_rotateleft"
                    reshape_node = helper.make_node(
                        "Reshape",
                        inputs=[node.input[0], reshape_shape_name],
                        outputs=[reshape_output_name],
                        name=node.name + "_rotateleft",
                    )
                    nodes_to_add.append(reshape_node)

                    # Rewire InstanceNormalization input
                    node.input[0] = reshape_output_name

    for node in nodes_to_add:
        graph.node.append(node)

    logger.info(
        "Inserted %d Reshape nodes before InstanceNormalization nodes with channel/scale mismatch.", len(nodes_to_add)
    )


def add_value_info(graph, tensor_name, dtype=TensorProto.FLOAT, shape=None):
    vi = helper.make_tensor_value_info(tensor_name, dtype, shape)
    graph.value_info.append(vi)


def transform_decompose_lstm(model):
    """Decompose ONNX LSTM nodes into a Loop with basic ONNX ops (MatMul, Add, Sigmoid, Tanh, Mul, etc.)

    Only supports single-layer, unidirectional LSTM for clarity.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    lstm_counter = 0

    for node in list(graph.node):
        if node.op_type == "LSTM":
            X, W, R, B = node.input[:4]
            initial_h = node.input[5]
            initial_c = node.input[6]

            # Get hidden_size and input_size from W shape
            W_arr = None
            hidden_size = None
            for init in graph.initializer:
                if init.name == W:
                    W_arr = numpy_helper.to_array(init)
                    hidden_size = W_arr.shape[1] // 4
                    break

            # Robustly get seq_length and batch_size from X shape
            seq_length = None
            batch_size = None
            for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
                if value_info.name == X:
                    dims = value_info.type.tensor_type.shape.dim
                    if len(dims) >= 2:
                        seq_length = dims[0].dim_value
                        batch_size = dims[1].dim_value
                    break
            if seq_length is None or batch_size is None:
                raise RuntimeError(f"Could not determine seq_length or batch_size for input {X}")

            # Add initializer for value 1
            one_initializer_name = f"one_val_{lstm_counter}"
            one_initializer = numpy_helper.from_array(np.array([1], dtype=np.int64), name=one_initializer_name)
            graph.initializer.append(one_initializer)

            zero_initializer_name = f"zero_val_{lstm_counter}"
            zero_initializer = numpy_helper.from_array(np.array([0], dtype=np.int64), name=zero_initializer_name)
            graph.initializer.append(zero_initializer)

            # Create initializer for axes_0
            axes_0_name = f"axes_0_{lstm_counter}"
            axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name=axes_0_name)
            graph.initializer.append(axes_0)

            # Create initializer for axes
            ax_name = f"axes_{lstm_counter}"
            ax = numpy_helper.from_array(np.array([0], dtype=np.int64), name=ax_name)
            graph.initializer.append(ax)

            # Create initializer for steps
            ste_name = f"steps_{lstm_counter}"
            ste = numpy_helper.from_array(np.array([1], dtype=np.int64), name=ste_name)
            graph.initializer.append(ste)

            x_t_name = f"x_t_{lstm_counter}"
            x_t_slice = helper.make_node(
                "Slice",
                [X, zero_initializer_name, one_initializer_name, ax_name, ste_name],
                [x_t_name],
                name=f"slice_x_t_{lstm_counter}",
            )

            # Split W, R, B for gates
            split_4h_name = f"split_4h_{lstm_counter}"
            split_4h = helper.make_node(
                "Constant",
                [],
                [split_4h_name],
                value=helper.make_tensor(
                    f"split_4h_val_{lstm_counter}",
                    TensorProto.INT64,
                    [4],
                    [hidden_size, hidden_size, hidden_size, hidden_size],
                ),
                name=f"const_split_4h_{lstm_counter}",
            )
            split_8h_name = f"split_8h_{lstm_counter}"
            split_8h = helper.make_node(
                "Constant",
                [],
                [split_8h_name],
                value=helper.make_tensor(
                    f"split_8h_val_{lstm_counter}", TensorProto.INT64, [2], [4 * hidden_size, 4 * hidden_size]
                ),
                name=f"const_split_8h_{lstm_counter}",
            )
            split_W = helper.make_node(
                "Split",
                [W, split_4h_name],
                [f"W_i_{lstm_counter}", f"W_o_{lstm_counter}", f"W_f_{lstm_counter}", f"W_c_{lstm_counter}"],
                axis=2,
                name=f"split_W_{lstm_counter}",
            )
            split_R = helper.make_node(
                "Split",
                [R, split_4h_name],
                [f"R_i_{lstm_counter}", f"R_o_{lstm_counter}", f"R_f_{lstm_counter}", f"R_c_{lstm_counter}"],
                axis=2,
                name=f"split_R_{lstm_counter}",
            )
            split_B = helper.make_node(
                "Split",
                [B, split_8h_name],
                [f"B_W_{lstm_counter}", f"B_R_{lstm_counter}"],
                axis=3,
                name=f"split_B_{lstm_counter}",
            )
            split_B_W = helper.make_node(
                "Split",
                [f"B_W_{lstm_counter}", split_4h_name],
                [f"b_Wi_{lstm_counter}", f"b_Wo_{lstm_counter}", f"b_Wf_{lstm_counter}", f"b_Wc_{lstm_counter}"],
                axis=3,
                name=f"split_B_W_{lstm_counter}",
            )
            split_B_R = helper.make_node(
                "Split",
                [f"B_R_{lstm_counter}", split_4h_name],
                [f"b_Ri_{lstm_counter}", f"b_Ro_{lstm_counter}", f"b_Rf_{lstm_counter}", f"b_Rc_{lstm_counter}"],
                axis=3,
                name=f"split_B_R_{lstm_counter}",
            )

            # Add Transpose nodes for weights before MatMul
            transpose_Wi = helper.make_node(
                "Transpose",
                [f"W_i_{lstm_counter}"],
                [f"W_i_T_{lstm_counter}"],
                name=f"transpose_Wi_{lstm_counter}",
                perm=[0, 2, 1],
            )
            transpose_Wf = helper.make_node(
                "Transpose",
                [f"W_f_{lstm_counter}"],
                [f"W_f_T_{lstm_counter}"],
                name=f"transpose_Wf_{lstm_counter}",
                perm=[0, 2, 1],
            )
            transpose_Wo = helper.make_node(
                "Transpose",
                [f"W_o_{lstm_counter}"],
                [f"W_o_T_{lstm_counter}"],
                name=f"transpose_Wo_{lstm_counter}",
                perm=[0, 2, 1],
            )
            transpose_Wc = helper.make_node(
                "Transpose",
                [f"W_c_{lstm_counter}"],
                [f"W_c_T_{lstm_counter}"],
                name=f"transpose_Wc_{lstm_counter}",
                perm=[0, 2, 1],
            )
            transpose_Ri = helper.make_node(
                "Transpose",
                [f"R_i_{lstm_counter}"],
                [f"R_i_T_{lstm_counter}"],
                name=f"transpose_Ri_{lstm_counter}",
                perm=[0, 2, 1],
            )
            transpose_Rf = helper.make_node(
                "Transpose",
                [f"R_f_{lstm_counter}"],
                [f"R_f_T_{lstm_counter}"],
                name=f"transpose_Rf_{lstm_counter}",
                perm=[0, 2, 1],
            )
            transpose_Ro = helper.make_node(
                "Transpose",
                [f"R_o_{lstm_counter}"],
                [f"R_o_T_{lstm_counter}"],
                name=f"transpose_Ro_{lstm_counter}",
                perm=[0, 2, 1],
            )
            transpose_Rc = helper.make_node(
                "Transpose",
                [f"R_c_{lstm_counter}"],
                [f"R_c_T_{lstm_counter}"],
                name=f"transpose_Rc_{lstm_counter}",
                perm=[0, 2, 1],
            )

            # Gates: i, f, o, g
            # Input gate
            xW_i = helper.make_node(
                "MatMul",
                [x_t_name, f"W_i_T_{lstm_counter}"],
                [f"xW_i_{lstm_counter}"],
                name=f"matmul_xW_i_{lstm_counter}",
            )
            hR_i = helper.make_node(
                "MatMul",
                [initial_h, f"R_i_T_{lstm_counter}"],
                [f"hR_i_{lstm_counter}"],
                name=f"matmul_hR_i_{lstm_counter}",
            )
            b_i = helper.make_node(
                "Add",
                [f"b_Wi_{lstm_counter}", f"b_Ri_{lstm_counter}"],
                [f"b_i_{lstm_counter}"],
                name=f"add_b_i_{lstm_counter}",
            )
            i_t_pre = helper.make_node(
                "Add",
                [f"xW_i_{lstm_counter}", f"hR_i_{lstm_counter}"],
                [f"i_t_pre1_{lstm_counter}"],
                name=f"add_i_t_pre1_{lstm_counter}",
            )
            i_t_pre2 = helper.make_node(
                "Add",
                [f"i_t_pre1_{lstm_counter}", f"b_i_{lstm_counter}"],
                [f"i_t_pre2_{lstm_counter}"],
                name=f"add_i_t_pre2_{lstm_counter}",
            )
            i_t = helper.make_node(
                "Sigmoid", [f"i_t_pre2_{lstm_counter}"], [f"i_t_{lstm_counter}"], name=f"sigmoid_i_t_{lstm_counter}"
            )
            # Forget gate
            xW_f = helper.make_node(
                "MatMul",
                [x_t_name, f"W_f_T_{lstm_counter}"],
                [f"xW_f_{lstm_counter}"],
                name=f"matmul_xW_f_{lstm_counter}",
            )
            hR_f = helper.make_node(
                "MatMul",
                [initial_h, f"R_f_T_{lstm_counter}"],
                [f"hR_f_{lstm_counter}"],
                name=f"matmul_hR_f_{lstm_counter}",
            )
            b_f = helper.make_node(
                "Add",
                [f"b_Wf_{lstm_counter}", f"b_Rf_{lstm_counter}"],
                [f"b_f_{lstm_counter}"],
                name=f"add_b_f_{lstm_counter}",
            )
            f_t_pre = helper.make_node(
                "Add",
                [f"xW_f_{lstm_counter}", f"hR_f_{lstm_counter}"],
                [f"f_t_pre1_{lstm_counter}"],
                name=f"add_f_t_pre1_{lstm_counter}",
            )
            f_t_pre2 = helper.make_node(
                "Add",
                [f"f_t_pre1_{lstm_counter}", f"b_f_{lstm_counter}"],
                [f"f_t_pre2_{lstm_counter}"],
                name=f"add_f_t_pre2_{lstm_counter}",
            )
            f_t = helper.make_node(
                "Sigmoid", [f"f_t_pre2_{lstm_counter}"], [f"f_t_{lstm_counter}"], name=f"sigmoid_f_t_{lstm_counter}"
            )
            # Output gate
            xW_o = helper.make_node(
                "MatMul",
                [x_t_name, f"W_o_T_{lstm_counter}"],
                [f"xW_o_{lstm_counter}"],
                name=f"matmul_xW_o_{lstm_counter}",
            )
            hR_o = helper.make_node(
                "MatMul",
                [initial_h, f"R_o_T_{lstm_counter}"],
                [f"hR_o_{lstm_counter}"],
                name=f"matmul_hR_o_{lstm_counter}",
            )
            b_o = helper.make_node(
                "Add",
                [f"b_Wo_{lstm_counter}", f"b_Ro_{lstm_counter}"],
                [f"b_o_{lstm_counter}"],
                name=f"add_b_o_{lstm_counter}",
            )
            o_t_pre = helper.make_node(
                "Add",
                [f"xW_o_{lstm_counter}", f"hR_o_{lstm_counter}"],
                [f"o_t_pre1_{lstm_counter}"],
                name=f"add_o_t_pre1_{lstm_counter}",
            )
            o_t_pre2 = helper.make_node(
                "Add",
                [f"o_t_pre1_{lstm_counter}", f"b_o_{lstm_counter}"],
                [f"o_t_pre2_{lstm_counter}"],
                name=f"add_o_t_pre2_{lstm_counter}",
            )
            o_t = helper.make_node(
                "Sigmoid", [f"o_t_pre2_{lstm_counter}"], [f"o_t_{lstm_counter}"], name=f"sigmoid_o_t_{lstm_counter}"
            )
            # Cell gate
            xW_c = helper.make_node(
                "MatMul",
                [x_t_name, f"W_c_T_{lstm_counter}"],
                [f"xW_c_{lstm_counter}"],
                name=f"matmul_xW_c_{lstm_counter}",
            )
            hR_c = helper.make_node(
                "MatMul",
                [initial_h, f"R_c_T_{lstm_counter}"],
                [f"hR_c_{lstm_counter}"],
                name=f"matmul_hR_c_{lstm_counter}",
            )
            b_c = helper.make_node(
                "Add",
                [f"b_Wc_{lstm_counter}", f"b_Rc_{lstm_counter}"],
                [f"b_c_{lstm_counter}"],
                name=f"add_b_c_{lstm_counter}",
            )
            g_t_pre = helper.make_node(
                "Add",
                [f"xW_c_{lstm_counter}", f"hR_c_{lstm_counter}"],
                [f"g_t_pre1_{lstm_counter}"],
                name=f"add_g_t_pre1_{lstm_counter}",
            )
            g_t_pre2 = helper.make_node(
                "Add",
                [f"g_t_pre1_{lstm_counter}", f"b_c_{lstm_counter}"],
                [f"g_t_pre2_{lstm_counter}"],
                name=f"add_g_t_pre2_{lstm_counter}",
            )
            g_t = helper.make_node(
                "Tanh", [f"g_t_pre2_{lstm_counter}"], [f"g_t_{lstm_counter}"], name=f"tanh_g_t_{lstm_counter}"
            )

            f_c_prev = helper.make_node(
                "Mul",
                [f"f_t_{lstm_counter}", initial_c],
                [f"f_c_prev_{lstm_counter}"],
                name=f"mul_f_c_prev_{lstm_counter}",
            )
            i_g = helper.make_node(
                "Mul",
                [f"i_t_{lstm_counter}", f"g_t_{lstm_counter}"],
                [f"i_g_{lstm_counter}"],
                name=f"mul_i_g_{lstm_counter}",
            )
            c_t = helper.make_node(
                "Add",
                [f"f_c_prev_{lstm_counter}", f"i_g_{lstm_counter}"],
                [node.output[2]],
                name=f"add_c_t_{lstm_counter}",
            )
            tanh_c_t = helper.make_node(
                "Tanh", [node.output[2]], [f"tanh_c_t_{lstm_counter}"], name=f"tanh_c_t_{lstm_counter}"
            )
            h_t = helper.make_node(
                "Mul",
                [f"o_t_{lstm_counter}", f"tanh_c_t_{lstm_counter}"],
                [node.output[0]],
                name=f"mul_h_t_{lstm_counter}",
            )
            indentity_i = helper.make_node(
                "Identity", [node.output[0]], [node.output[1]], name=f"identity_i_{lstm_counter}"
            )

            # Add constant one initializer
            graph.initializer.append(
                numpy_helper.from_array(np.array(1, dtype=np.int64), name=node.name + f"_one_{lstm_counter}")
            )

            nodes_to_remove.append(node)
            nodes_to_add.extend(
                [
                    x_t_slice,
                    split_4h,
                    split_8h,
                    split_W,
                    split_R,
                    split_B,
                    split_B_W,
                    split_B_R,
                    transpose_Wi,
                    transpose_Wf,
                    transpose_Wo,
                    transpose_Wc,
                    transpose_Ri,
                    transpose_Rf,
                    transpose_Ro,
                    transpose_Rc,
                    xW_i,
                    hR_i,
                    b_i,
                    i_t_pre,
                    i_t_pre2,
                    i_t,
                    xW_f,
                    hR_f,
                    b_f,
                    f_t_pre,
                    f_t_pre2,
                    f_t,
                    xW_o,
                    hR_o,
                    b_o,
                    o_t_pre,
                    o_t_pre2,
                    o_t,
                    xW_c,
                    hR_c,
                    b_c,
                    g_t_pre,
                    g_t_pre2,
                    g_t,
                    f_c_prev,
                    i_g,
                    c_t,
                    tanh_c_t,
                    h_t,
                    indentity_i,
                ]
            )
            lstm_counter += 1

    for node in nodes_to_remove:
        graph.node.remove(node)
    for node in nodes_to_add:
        graph.node.append(node)
    return model


def transform_reshape_clip_reducesum_manual(model):
    from onnx import TensorProto, helper

    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    counter = 0
    for node in list(graph.node):
        if node.op_type == "Reshape":
            reshape_out = node.output[0]
            # Find Clip
            clip_nodes = [n for n in graph.node if n.input and n.input[0] == reshape_out and n.op_type == "Clip"]
            if not clip_nodes:
                continue
            clip_node = clip_nodes[0]
            clip_out = clip_node.output[0]
            # Find ReduceSum
            rs_nodes = [n for n in graph.node if n.input and n.input[0] == clip_out and n.op_type == "ReduceSum"]
            if not rs_nodes:
                continue
            rs_node = rs_nodes[0]

            # Remove old nodes
            nodes_to_remove.extend([node, clip_node, rs_node])

            # Get parameters
            input_tensor = node.input[0]  # Use the input to the original Reshape
            clip_min = clip_node.input[1] if len(clip_node.input) > 1 else None
            clip_max = clip_node.input[2] if len(clip_node.input) > 2 else None
            axes = rs_node.input[1] if len(rs_node.input) > 1 else None
            output_name = rs_node.output[0]

            # Create slice parameters (example: split along axis 3, first half and second half)
            slice_0_starts = helper.make_tensor(
                name=f"slice_0_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[0]
            )
            slice_0_ends = helper.make_tensor(
                name=f"slice_0_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[4]
            )
            slice_1_starts = helper.make_tensor(
                name=f"slice_1_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[4]
            )
            slice_1_ends = helper.make_tensor(
                name=f"slice_1_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[8]
            )
            slice_axes = helper.make_tensor(
                name=f"slice_axes_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[3]
            )

            graph.initializer.extend([slice_0_starts, slice_0_ends, slice_1_starts, slice_1_ends, slice_axes])

            # Unique names for all new nodes
            slice0_out = f"slice0_out_{counter}"
            clip0_out = f"clip0_out_{counter}"
            rs0_out = f"rs0_out_{counter}"
            slice1_out = f"slice1_out_{counter}"
            clip1_out = f"clip1_out_{counter}"
            rs1_out = f"rs1_out_{counter}"

            # Slice 0 branch
            slice0 = helper.make_node(
                "Slice",
                [input_tensor, f"slice_0_starts_{counter}", f"slice_0_ends_{counter}", f"slice_axes_{counter}"],
                [slice0_out],
            )
            clip0_inputs = [slice0_out]
            if clip_min is not None:
                clip0_inputs.append(clip_min)
            if clip_max is not None:
                clip0_inputs.append(clip_max)
            clip0 = helper.make_node("Clip", clip0_inputs, [clip0_out])
            rs0 = helper.make_node("ReduceSum", [clip0_out, axes], [rs0_out])
            # Slice 1 branch
            slice1 = helper.make_node(
                "Slice",
                [input_tensor, f"slice_1_starts_{counter}", f"slice_1_ends_{counter}", f"slice_axes_{counter}"],
                [slice1_out],
            )
            clip1_inputs = [slice1_out]
            if clip_min is not None:
                clip1_inputs.append(clip_min)
            if clip_max is not None:
                clip1_inputs.append(clip_max)
            clip1 = helper.make_node("Clip", clip1_inputs, [clip1_out])
            rs1 = helper.make_node("ReduceSum", [clip1_out, axes], [rs1_out])
            # Concat
            concat = helper.make_node("Concat", [rs0_out, rs1_out], [output_name], axis=3)

            nodes_to_add.extend([slice0, clip0, rs0, slice1, clip1, rs1, concat])
            counter += 1

    logger.info("Updated %d Reshape nodes with Clip ReduceSum", counter)
    # Remove old nodes
    for n in nodes_to_remove:
        graph.node.remove(n)
    # Add new nodes
    for n in nodes_to_add:
        graph.node.append(n)
    return model


def transform_reshape_clip_transpose_clip_reshape(model):
    """Find pattern: Reshape -> Clip -> Transpose -> Clip -> Reshape
    Replace with: Slice into 6 tensors along first dim, reshape each to 4x4x115x199,
    transpose to 115x4x199x4, reshape to 1x115x4x796, concat all 6 along first axis
    to get 6x115x4x796, then reshape to 1x6x460x796
    """
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    counter = 0

    for node in list(graph.node):
        if node.op_type == "Reshape":
            # Check if first reshape shape is 5D immediately
            reshape_shape_initializer = None
            for init in graph.initializer:
                if init.name == node.input[1]:  # shape parameter of first reshape
                    reshape_shape_initializer = init
                    break

            if reshape_shape_initializer is None:
                continue

            # Check if shape is 5D
            shape_values = numpy_helper.to_array(reshape_shape_initializer)
            if len(shape_values) != 5:
                continue

            reshape1_out = node.output[0]

            # Find first Clip
            clip1_nodes = [n for n in graph.node if n.input and n.input[0] == reshape1_out and n.op_type == "Clip"]
            if not clip1_nodes:
                continue
            clip1_node = clip1_nodes[0]
            clip1_out = clip1_node.output[0]

            # Find Transpose
            transpose_nodes = [
                n for n in graph.node if n.input and n.input[0] == clip1_out and n.op_type == "Transpose"
            ]
            if not transpose_nodes:
                continue
            transpose_node = transpose_nodes[0]
            transpose_out = transpose_node.output[0]

            # Find second Clip
            clip2_nodes = [n for n in graph.node if n.input and n.input[0] == transpose_out and n.op_type == "Clip"]
            if not clip2_nodes:
                continue
            clip2_node = clip2_nodes[0]
            clip2_out = clip2_node.output[0]

            # Find second Reshape
            reshape2_nodes = [n for n in graph.node if n.input and n.input[0] == clip2_out and n.op_type == "Reshape"]
            if not reshape2_nodes:
                continue
            reshape2_node = reshape2_nodes[0]

            # Remove all old nodes
            nodes_to_remove.extend([node, clip1_node, transpose_node, clip2_node, reshape2_node])

            # Get parameters
            input_tensor = node.input[0]  # Use the input to the first Reshape
            output_name = reshape2_node.output[0]

            # Create slice parameters for 6 slices along first dimension
            slice_starts = []
            slice_ends = []
            slice_axes = helper.make_tensor(
                name=f"slice_axes_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[0]
            )

            for i in range(6):
                slice_starts.append(
                    helper.make_tensor(
                        name=f"slice_{i}_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[i]
                    )
                )
                slice_ends.append(
                    helper.make_tensor(
                        name=f"slice_{i}_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[i + 1]
                    )
                )

            graph.initializer.extend([slice_axes] + slice_starts + slice_ends)

            # Create reshape shapes
            reshape_4d_shape = helper.make_tensor(
                name=f"reshape_4d_shape_{counter}", data_type=TensorProto.INT64, dims=[4], vals=[4, 4, 115, 199]
            )
            reshape_1d_shape = helper.make_tensor(
                name=f"reshape_1d_shape_{counter}", data_type=TensorProto.INT64, dims=[4], vals=[1, 1, 460, 796]
            )
            final_reshape_shape = helper.make_tensor(
                name=f"final_reshape_shape_{counter}", data_type=TensorProto.INT64, dims=[4], vals=[1, 6, 460, 796]
            )

            graph.initializer.extend([reshape_4d_shape, reshape_1d_shape, final_reshape_shape])

            # Process each of the 6 slices
            slice_outputs = []
            for i in range(6):
                # Slice
                slice_out = f"slice_{i}_out_{counter}"
                slice_node = helper.make_node(
                    "Slice",
                    [input_tensor, f"slice_{i}_starts_{counter}", f"slice_{i}_ends_{counter}", f"slice_axes_{counter}"],
                    [slice_out],
                    name=f"slice_{i}_transformed_{counter}",
                )

                # Reshape to 4x4x115x199
                reshape_4d_out = f"reshape_4d_{i}_out_{counter}"
                reshape_4d_node = helper.make_node(
                    "Reshape",
                    [slice_out, f"reshape_4d_shape_{counter}"],
                    [reshape_4d_out],
                    name=f"reshape_4d_{i}_transformed_{counter}",
                )

                # Transpose to 115x4x199x4
                transpose_out = f"transpose_{i}_out_{counter}"
                transpose_node = helper.make_node(
                    "Transpose",
                    [reshape_4d_out],
                    [transpose_out],
                    perm=[2, 0, 3, 1],
                    name=f"transpose_{i}_transformed_{counter}",
                )

                # Reshape to 1x115x4x796
                reshape_1d_out = f"reshape_1d_{i}_out_{counter}"
                reshape_1d_node = helper.make_node(
                    "Reshape",
                    [transpose_out, f"reshape_1d_shape_{counter}"],
                    [reshape_1d_out],
                    name=f"reshape_1d_{i}_transformed_{counter}",
                )

                slice_outputs.append(reshape_1d_out)
                nodes_to_add.extend([slice_node, reshape_4d_node, transpose_node, reshape_1d_node])

            # Concat all 6 slices along first axis to get 6x115x4x796
            concat_node = helper.make_node(
                "Concat", slice_outputs, [output_name], axis=1, name=f"concat_transformed_{counter}"
            )

            nodes_to_add.append(concat_node)
            counter += 1

    logger.info("Updated %d Reshape->Clip->Transpose->Clip->Reshape patterns with slice-based transformation", counter)

    # Remove old nodes
    for n in nodes_to_remove:
        graph.node.remove(n)

    # Add new nodes
    for n in nodes_to_add:
        graph.node.append(n)

    return model


def transform_reshape_transpose_reshape(model):
    """Find pattern: Reshape -> Transpose -> Reshape
    Replace with: Slice into 6 tensors along first dim, reshape each to 4x4x115x199,
    transpose to 115x4x199x4, reshape to 1x1x460x796, concat all 6 along axis 1
    """
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    counter = 0
    for node in list(graph.node):
        if node.op_type == "Reshape":
            # Skip if node name contains "_transformed"
            if "_transformed" in node.name:
                continue

            # Check if first reshape shape is 5D immediately
            reshape_shape_initializer = None
            for init in graph.initializer:
                if init.name == node.input[1]:  # shape parameter of first reshape
                    reshape_shape_initializer = init
                    break

            if reshape_shape_initializer is None:
                continue

            # Check if shape is 5D
            shape_values = numpy_helper.to_array(reshape_shape_initializer)
            if len(shape_values) != 5:
                continue

            reshape1_out = node.output[0]
            # Find Transpose
            transpose_nodes = [
                n for n in graph.node if n.input and n.input[0] == reshape1_out and n.op_type == "Transpose"
            ]
            if not transpose_nodes:
                continue
            transpose_node = transpose_nodes[0]
            # Skip if transpose node name contains "_transformed"
            if "_transformed" in transpose_node.name:
                continue
            transpose_out = transpose_node.output[0]
            # Find second Reshape
            reshape2_nodes = [
                n for n in graph.node if n.input and n.input[0] == transpose_out and n.op_type == "Reshape"
            ]
            if not reshape2_nodes:
                continue
            reshape2_node = reshape2_nodes[0]
            # Skip if reshape2 node name contains "_transformed"
            if "_transformed" in reshape2_node.name:
                continue
            # Remove all old nodes
            nodes_to_remove.extend([node, transpose_node, reshape2_node])
            # Get parameters
            input_tensor = node.input[0]  # Use the input to the first Reshape
            output_name = reshape2_node.output[0]
            # Create slice parameters for 6 slices along first dimension
            slice_starts = []
            slice_ends = []
            slice_axes = helper.make_tensor(
                name=f"slice_axes_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[0]
            )
            for i in range(6):
                slice_starts.append(
                    helper.make_tensor(
                        name=f"slice_{i}_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[i]
                    )
                )
                slice_ends.append(
                    helper.make_tensor(
                        name=f"slice_{i}_ends_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[i + 1]
                    )
                )
            graph.initializer.extend([slice_axes] + slice_starts + slice_ends)
            # Create reshape shapes
            reshape_4d_shape = helper.make_tensor(
                name=f"reshape_4d_shape_{counter}", data_type=TensorProto.INT64, dims=[4], vals=[4, 4, 115, 199]
            )
            reshape_1d_shape = helper.make_tensor(
                name=f"reshape_1d_shape_{counter}", data_type=TensorProto.INT64, dims=[4], vals=[1, 1, 460, 796]
            )
            graph.initializer.extend([reshape_4d_shape, reshape_1d_shape])
            # Process each of the 6 slices
            slice_outputs = []
            for i in range(6):
                # Slice
                slice_out = f"slice_{i}_out_{counter}"
                slice_node = helper.make_node(
                    "Slice",
                    [input_tensor, f"slice_{i}_starts_{counter}", f"slice_{i}_ends_{counter}", f"slice_axes_{counter}"],
                    [slice_out],
                )
                # Reshape to 4x4x115x199
                reshape_4d_out = f"reshape_4d_{i}_out_{counter}"
                reshape_4d_node = helper.make_node(
                    "Reshape", [slice_out, f"reshape_4d_shape_{counter}"], [reshape_4d_out]
                )
                # Transpose to 115x4x199x4
                transpose_out = f"transpose_{i}_out_{counter}"
                transpose_node = helper.make_node("Transpose", [reshape_4d_out], [transpose_out], perm=[2, 0, 3, 1])
                # Reshape to 1x1x460x796
                reshape_1d_out = f"reshape_1d_{i}_out_{counter}"
                reshape_1d_node = helper.make_node(
                    "Reshape", [transpose_out, f"reshape_1d_shape_{counter}"], [reshape_1d_out]
                )
                slice_outputs.append(reshape_1d_out)
                nodes_to_add.extend([slice_node, reshape_4d_node, transpose_node, reshape_1d_node])
            # Concat all 6 slices along axis 1 to get 6x1x460x796
            concat_node = helper.make_node("Concat", slice_outputs, [output_name], axis=1)
            nodes_to_add.append(concat_node)
            counter += 1
    logger.info("Updated %d Reshape->Transpose->Reshape patterns with slice-based transformation", counter)
    # Remove old nodes
    for n in nodes_to_remove:
        graph.node.remove(n)
    # Add new nodes
    for n in nodes_to_add:
        graph.node.append(n)
    return model
