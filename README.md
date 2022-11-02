# LEVEL1 P Stage - Book Recommendation


## RecSys_4


|고현욱|김동건|김현희|민복기|용희원|
|---|---|---|---|---|
|T4007|T4026|T4061|T4074|T4130|
|🫶🏻|🫶🏻|🫶🏻|🫶🏻|🫶🏻|

### train records

|File Name|model|epochs|batchsize|rmse|trainer|etc|
|---|---|---|---|---|---|---|
||||||||
||||||||

### 🚂 python main.py -- Options

🫶🏻 나중에 폴더 안으로 들어갈 부분..
####  Basic / Train Options

|Basic Option|Description |
|---|---|
|--DATA_PATH|Data Path 설정 |
|--MODEL|학습 및 예측할 모델 선택 |
|--DATA_SHUFFLE|데이터 셔플 여부를 조정 |
|--TEST_SIZE|Train/Valid split 비율을 조정 |

|Training Option|Description |
|---|---|
|--BATCH_SIZE|Batch size를 조정 |
|--EPOCHS|Epoch 수를 조정 |
|--LR|Learning Rate를 조정 |
|--WEIGHT_DECAY|Adam optimizer에서 정규화에 사용하는 값을 조정 |

#### MODEL OPTIONS

|FM Option|Description |
|---|---|
|--FM_EMBED_DIM|FM에서 embedding시킬 차원을 조정 |

|FFM Option|Description |
|---|---|
|--FFM_EMBED_DIM|FFM에서 embedding시킬 차원을 조정 |

|NCF Option|Description |
|---|---|
|--NCF_EMBED_DIM|NCF에서 embedding시킬 차원을 조정 |
|--NCF_MLP_DIMS|NCF에서 MLP Network의 차원을 조정 |
|--NCF_DROPOUT|NCF에서 Dropout rate를 조정 |


|WDN Option|Description |
|---|---|
|--WDN_EMBED_DIM|WDN에서 embedding시킬 차원을 조정 |
|--WDN_MLP_DIMS|WDN에서 MLP Network의 차원을 조정 |
|--WDN_DROPOUT|WDN에서 Dropout rate를 조정 |

|DCN Option|Description |
|---|---|
|--DCN_EMBED_DIM|DCN에서 embedding시킬 차원을 조정 |
|--DCN_MLP_DIMS|DCN에서 MLP Network의 차원을 조정 |
|--DCN_DROPOUT|DCN에서 Dropout rate를 조정 |
|--DCN_NUM_LAYERS|DCN에서 Cross Network의 레이어 수를 조정 |

|CNN_FM Option|Description |
|---|---|
|--CNN_FM_EMBED_DIM|CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정 |
|--CNN_FM_LATENT_DIM|CNN_FM에서 user/item/image에 대한 latent 차원을 조정 |

|DeepCoNN Option|Description||
|---|---|---|
|--DEEPCONN_VECTOR_CREATE|DEEP_CONN에서 text vector 생성 여부를 조정|최초 학습에만 True로 설정|
|--DEEPCONN_EMBED_DIM|DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정||
|--DEEPCONN_LATENT_DIM|DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정||
|--DEEPCONN_CONV_1D_OUT_DIM|DEEP_CONN에서 1D conv의 출력 크기를 조정||
|--DEEPCONN_KERNEL_SIZE|DEEP_CONN에서 1D conv의 kernel 크기를 조정|kernel : 3|
|--DEEPCONN_WORD_DIM|DEEP_CONN에서 1D conv의 입력 크기를 조정|Word : 768|
|--DEEPCONN_OUT_DIM|DEEP_CONN에서 1D conv의 출력 크기를 조정||


