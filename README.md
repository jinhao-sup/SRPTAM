# SRPTAM
This is our TensorFlow implementation of SRPTAM.  
Please cite our paper if you use the code~  :)  

## Model Train (ML1M for example)  
`CUDA_VISIBLE_DEVICES=0 nohup python main.py train -p configs/ml1m.json`  

## Model Test (ML1M for example)  
`CUDA_VISIBLE_DEVICES=0 python main.py eval -p configs/ml1m.json`  
###  Model Test/eval log (ML1M for example)  
`2025-09-22 16:34:49.762789: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-09-22 16:34:51,048:INFO:srtam:exp/model/ml1m_10ucore_5icore/samples_step100/nepoch100/srtam_lr0.001_batch512_dim64_seqlen50_l2emb0.0_nblocks2_nheads2_dropout0.3_tempo-dim16-linspace8_residual-add_glob0.3_l2u0.0_l2i0.0_test_version3.2.1
2025-09-22 16:34:51,199:INFO:srtam:Number of users: 6040
2025-09-22 16:34:51,199:INFO:srtam:Number of items: 3416
2025-09-22 16:34:51,201:INFO:srtam:Number of interactions: 999611
2025-09-22 16:34:51,201:INFO:srtam:Density: 0.04845
2025-09-22 16:34:51.281870: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-09-22 16:34:51.867110: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22182 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, compute capability: 8.6
2025-09-22 16:34:51,873:INFO:srtam:Load <class 'models.srtam.SRTAM'> model from exp/model/ml1m_10ucore_5icore/samples_step100/nepoch100/srtam_lr0.001_batch512_dim64_seqlen50_l2emb0.0_nblocks2_nheads2_dropout0.3_tempo-dim16-linspace8_residual-add_glob0.3_l2u0.0_l2i0.0_test_version3.2.1
WARNING:tensorflow:From /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
2025-09-22 16:34:51,949:INFO:srtam:Scale input sequence
2025-09-22 16:34:52,180:INFO:srtam:Scale context sequences
2025-09-22 16:34:56.629935: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
2025-09-22 16:34:57,047:INFO:srtam:EVALUATION for #1 COHORT
Evaluating...:   0%|                                                                                                                                          | 0/9 [00:00<?, ?it/s]2025-09-22 16:34:58.485908: I tensorflow/stream_executor/cuda/cuda_blas.cc:1786] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2025-09-22 16:34:59.048512: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8101
Evaluating...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:04<00:00,  1.82it/s]
2025-09-22 16:35:02,034:INFO:srtam:Number of bad results: 0
2025-09-22 16:35:02,036:INFO:srtam:Step #1,NDCG@10  0.60422 ,HR@10  0.82422 ,MAP@10  0.53433 ,NDCG@5  0.57206 ,HR@5  0.72483 ,MAP@5  0.52104 
2025-09-22 16:35:02,036:INFO:srtam:EVALUATION for #2 COHORT
Evaluating...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:02<00:00,  3.74it/s]
2025-09-22 16:35:04,475:INFO:srtam:Number of bad results: 0
2025-09-22 16:35:04,476:INFO:srtam:Step #2,NDCG@10  0.60811 ,HR@10  0.82964 ,MAP@10  0.53780 ,NDCG@5  0.57488 ,HR@5  0.72700 ,MAP@5  0.52407 
2025-09-22 16:35:04,476:INFO:srtam:EVALUATION for #3 COHORT
Evaluating...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:02<00:00,  3.80it/s]
2025-09-22 16:35:06,883:INFO:srtam:Number of bad results: 0
2025-09-22 16:35:06,885:INFO:srtam:Step #3,NDCG@10  0.60094 ,HR@10  0.82422 ,MAP@10  0.52999 ,NDCG@5  0.56831 ,HR@5  0.72374 ,MAP@5  0.51644 
2025-09-22 16:35:06,885:INFO:srtam:EVALUATION for #4 COHORT
Evaluating...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:02<00:00,  3.82it/s]
2025-09-22 16:35:09,274:INFO:srtam:Number of bad results: 0
2025-09-22 16:35:09,276:INFO:srtam:Step #4,NDCG@10  0.60215 ,HR@10  0.82096 ,MAP@10  0.53258 ,NDCG@5  0.56956 ,HR@5  0.72114 ,MAP@5  0.51894 
2025-09-22 16:35:09,276:INFO:srtam:EVALUATION for #5 COHORT
Evaluating...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:02<00:00,  3.84it/s]
2025-09-22 16:35:11,649:INFO:srtam:Number of bad results: 0
2025-09-22 16:35:11,650:INFO:srtam:Step #5,NDCG@10  0.60180 ,HR@10  0.82552 ,MAP@10  0.53080 ,NDCG@5  0.56827 ,HR@5  0.72244 ,MAP@5  0.51685 
2025-09-22 16:35:11,651:INFO:srtam:RESULTS:
NDCG@10:  0.60344 +/-  0.00257
HR@10:  0.82491 +/-  0.00280
MAP@10:  0.53310 +/-  0.00279
NDCG@5:  0.57061 +/-  0.00254
HR@5:  0.72383 +/-  0.00201
MAP@5:  0.51947 +/-  0.00283
`


