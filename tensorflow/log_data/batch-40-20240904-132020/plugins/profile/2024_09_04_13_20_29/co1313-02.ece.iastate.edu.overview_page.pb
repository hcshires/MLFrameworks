�   *	�Zd;A2U
Iterator::Root::ParallelMapV2�$�����^@!�	��I@)$�����^@1�	��I@:Preprocessing2O
Iterator::Root::BatchV2����U�qM@!�4Z�k�8@)��g��mH@1�Zm��4@:Preprocessing2b
*Iterator::Root::BatchV2::Prefetch::Shuffle�N	��8�d@@!M�:�+@)4f���6@1�Y�TX#@:Preprocessing2F
Iterator::Root�g�gh@!�[�q��T@)��,��$+@1�t���@:Preprocessing2Y
!Iterator::Root::BatchV2::Prefetch�NΊ��>$@!�hm��@)Ί��>$@1�hm��@:Preprocessing2l
4Iterator::Root::BatchV2::Prefetch::Shuffle::Prefetch�N�_�n�#@!!q�phw@)�_�n�#@1!q�phw@:Preprocessing2}
EIterator::Root::BatchV2::Prefetch::Shuffle::Prefetch::MemoryCacheImpl�N�QI���@!>��y@)�QI���@1>��y@:Preprocessing2y
AIterator::Root::BatchV2::Prefetch::Shuffle::Prefetch::MemoryCache�N��S!@!
#ĥ@)�hUM�
@1���!X�?:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch9���?!KT!��{�?)9���?1KT!��{�?:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapmt�Oq��?!����]�?)��ٮ��?1���ͣ��?:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map1�߄B�?!B���GO�?):<��Ӹ�?1�P��Q�?:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipX�^���?!fQ}��?)�N��C�?1�L�^#��?:Preprocessing2�
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatQ�v0b��?!�
����?)�J�*n�?1,qx��?:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���"��?!g���S��?)�̰Q�o�?1�m�y߂?:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat��u�|Ϙ?!/!(ؚބ?)q<��f�?1��:/U)�?:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice䠄���?!`�8��z?)䠄���?1`�8��z?:Preprocessing2�
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range؜�gBs?!��H�?3`?)؜�gBs?1��H�?3`?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�5Y��q?!��iG-�]?)�5Y��q?1��iG-�]?:Preprocessing2�
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorͮ{+S?!�N�D�
@?)ͮ{+S?1�N�D�
@?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q�����>@"�
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb�30.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Tco1313-02.ece.iastate.edu: Failed to load libcupti (is it installed and accessible?)