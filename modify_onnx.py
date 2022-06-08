import onnx
from onnx import helper, checker
from onnx import TensorProto
import re
import argparse
import pdb

output_model = 'testv3.onnx'

def createGraphMemberMap(graph_member_list):
    member_map=dict()
    for n in graph_member_list:
        member_map[n.name]=n
    return member_map

onnx_model = onnx.load('test.onnx')
graph = onnx_model.graph
node_map = createGraphMemberMap(graph.node)
input_map = createGraphMemberMap(graph.input)
output_map = createGraphMemberMap(graph.output)
# # pdb.set_trace()
# # print(node_map)
# nodes = graph.node
# # for node_id,node in enumerate(graph.node):
# #     if (node.op_type == "InstanceNormalization"):
# #         print("######%s######" % node_id)                                                                                                                                                                                                                                 
# #         print(node)                                                                   

# for i in range(len(nodes)):
#     if nodes[i].op_type == 'InstanceNormalization':
#         node_rise = nodes[i]

for name in node_map.keys():
    
    # if name == 'InstanceNormalization_532' or name == 'InstanceNormalization_538' or name == 'InstanceNormalization_544' or \
    #     name == 'InstanceNormalization_558' or name == 'InstanceNormalization_587' or name == 'InstanceNormalization_616' or \
    #     name == 'InstanceNormalization_653' or name == 'InstanceNormalization_659' or name == 'InstanceNormalization_665':
    #     # pdb.set_trace()
    #     graph.node.remove(node_map[name])

    if name == 'InstanceNormalization_532':
        # pdb.set_trace()
        graph.node.remove(node_map[name])

for name in node_map.keys():
    if name == "PRelu_534":
        node_rise = node_map[name]
        ii = node_rise.input
        node_rise.input[0] = '807'
    # if name == "PRelu_540":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '813'
    # if name == "PRelu_546":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '819'
    # if name == "PRelu_560":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '833' 
    # if name == "PRelu_589":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '862'  
    # if name == "PRelu_618":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '891'
    # if name == "PRelu_655":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '928'     
    # if name == "PRelu_661":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '934' 
    # if name == "PRelu_667":
    #     node_rise = node_map[name]
    #     ii = node_rise.input
    #     node_rise.input[0] = '940' 

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, output_model)

