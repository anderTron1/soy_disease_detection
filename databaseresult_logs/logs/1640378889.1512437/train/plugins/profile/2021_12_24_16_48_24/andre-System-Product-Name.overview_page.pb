?	????S@????S@!????S@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????S@V??#i??1??hq??R@AʉvR~??Iٔ+????rEagerKernelExecute 0*	t???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??Ց#>@!???%??X@)??Ց#>@1???%??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?8?????!h?7?????)?8?????1h?7?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??ypw֮?!:??{????)?Y??Bs??1R?G?_??:Preprocessing2F
Iterator::Model|~!<ڰ?!???'???)Ǻ???v?1?Ԓ?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap4GV~$>@!??4l?X@)?????`?19?r??|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noIP\7;K@Q;?LL?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V??#i??V??#i??!V??#i??      ??!       "	??hq??R@??hq??R@!??hq??R@*      ??!       2	ʉvR~??ʉvR~??!ʉvR~??:	ٔ+????ٔ+????!ٔ+????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qP\7;K@y;?LL?W@?"j
>gradient_tape/sequential/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????￫?!????￫?0"j
>gradient_tape/sequential/conv2d_13/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter_???a???!tNړ(???0"j
>gradient_tape/sequential/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter%???禫?!Ö?G????0"h
=gradient_tape/sequential/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput9z@????!Q?d.?4??0"h
=gradient_tape/sequential/conv2d_13/Conv2D/Conv2DBackpropInputConv2DBackpropInput|??M<???!p??A`???0"h
=gradient_tape/sequential/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput???8???!???0???0"g
<gradient_tape/sequential/conv2d_9/Conv2D/Conv2DBackpropInputConv2DBackpropInput?j??R+??!^??`L???0"g
<gradient_tape/sequential/conv2d_8/Conv2D/Conv2DBackpropInputConv2DBackpropInpute2?t??!??p????0"i
=gradient_tape/sequential/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterq^4Pcї?!k?s?	???0";
sequential/conv2d_11/Relu_FusedConv2De1f??I??!???ѥ???Q      Y@Y	?=???@aO#,?4bW@q7??]???yt?CdXf?"?	
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