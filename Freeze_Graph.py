import tensorflow._api.v2.compat.v1 as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

tf.disable_v2_behavior()

freeze_graph.freeze_graph(input_graph='weather_prediction.pbtxt',
                          input_saver='',
                          input_binary=True,
                          input_checkpoint='weather_prediction.ckpt',
                          output_node_names='y_output',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph='frozen_weather_prediction.pb',
                          clear_devices=True,
                          initializer_nodes='')

input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_weather_prediction.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                     ['x_input'],
                                                                     ['y_output'],
                                                                     tf.float32.as_datatype_enum)
f = tf.gfile.FastGFile('optimized_weather_prediction.pb', 'w')
f.write(output_graph_def.SerializeToString())

