?	??}Unw?@??}Unw?@!??}Unw?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??}Unw?@FA???@1?fwT?@A?J???I?l??p???rEagerKernelExecute 0*	gfff???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator^-wf?\X@!0?_.C?X@)^-wf?\X@10?_.C?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism#j??G??!q?+|޽??)????ne??1p/,????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchZ??? ͘?!qK+??o??)Z??? ͘?1qK+??o??:Preprocessing2F
Iterator::Model?????!?<?W4??)?O?I?5s?1I?ۖ˳s?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?DR?\X@!uy?X@)g???uj?1??0Q#k?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI????Z??Q7??_J?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	FA???@FA???@!FA???@      ??!       "	?fwT?@?fwT?@!?fwT?@*      ??!       2	?J????J???!?J???:	?l??p????l??p???!?l??p???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????Z??y7??_J?X@?"i
=gradient_tape/sequential/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!???????0"h
=gradient_tape/sequential/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput???|???!=U?Ѷ?0"\
1Adadelta/Adadelta/update_36/ResourceApplyAdadeltaResourceApplyAdadelta?0??=6??!f?ɐ#???"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter؍????!nB7J??0"j
>gradient_tape/sequential/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ޝ?!f?????0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?*BHO???!?G"ӱ|??0"i
=gradient_tape/sequential/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Xys????!?r?A???0"j
>gradient_tape/sequential/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterri??H??!?P?Hm???0"j
>gradient_tape/sequential/conv2d_13/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?94v(	??!"?2?????0"h
=gradient_tape/sequential/conv2d_13/Conv2D/Conv2DBackpropInputConv2DBackpropInput????????!\-???9??0Q      Y@Y
ps???@ad̫[X@q?DL<h??y?<?+?5?"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 