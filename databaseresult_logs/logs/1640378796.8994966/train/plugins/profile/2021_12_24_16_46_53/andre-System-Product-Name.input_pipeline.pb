	.Ui?k?T@.Ui?k?T@!.Ui?k?T@	????sw?????sw?!????sw?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL.Ui?k?T@gE?D???1~oӟ?#T@A?7??̒??I??/??L??Y??A??s?rEagerKernelExecute 0*	I+???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorM???oE@!?g???X@)M???oE@1?g???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism!<?8b-??!oJ??đ??)???B?i??1;?]?޴??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchS??.???!???)?n??)S??.???1???)?n??:Preprocessing2F
Iterator::Model??@??_??!b?F???)??6?ُt?1z?K???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??R%pE@!O?\?w?X@)???9]s?1??˯?9??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????sw?I ?9??	@Q??%`?/X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	gE?D???gE?D???!gE?D???      ??!       "	~oӟ?#T@~oӟ?#T@!~oӟ?#T@*      ??!       2	?7??̒???7??̒??!?7??̒??:	??/??L????/??L??!??/??L??B      ??!       J	??A??s???A??s?!??A??s?R      ??!       Z	??A??s???A??s?!??A??s?b      ??!       JGPUY????sw?b q ?9??	@y??%`?/X@