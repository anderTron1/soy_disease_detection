	????S@????S@!????S@      ??!       "?
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
	V??#i??V??#i??!V??#i??      ??!       "	??hq??R@??hq??R@!??hq??R@*      ??!       2	ʉvR~??ʉvR~??!ʉvR~??:	ٔ+????ٔ+????!ٔ+????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qP\7;K@y;?LL?W@