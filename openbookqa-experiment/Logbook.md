# V1

weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
learning_rate = 5e-4 # max learning rate
device_type = 'cpu'
epochs = 2
--
Training Epoch 0
Train Batch ------  0 / 309 Accuracy: 0 / 16 = 0.0 Loss: 5.626175403594971
Validation Accuracy: 0 / 500 = 0.0
Train Batch ------  10 / 309 Accuracy: 32 / 160 = 0.2 Loss: 3.220670461654663
Train Batch ------  20 / 309 Accuracy: 46 / 160 = 0.2875 Loss: 3.11411714553833
Train Batch ------  30 / 309 Accuracy: 33 / 160 = 0.20625 Loss: 2.899670362472534
Train Batch ------  40 / 309 Accuracy: 31 / 160 = 0.19375 Loss: 2.3247339725494385
Train Batch ------  50 / 309 Accuracy: 33 / 160 = 0.20625 Loss: 2.688093900680542
Validation Accuracy: 138 / 500 = 0.276
Train Batch ------  60 / 309 Accuracy: 34 / 160 = 0.2125 Loss: 1.911884069442749
Train Batch ------  70 / 309 Accuracy: 42 / 160 = 0.2625 Loss: 1.989655613899231
Train Batch ------  80 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 1.9309368133544922
Train Batch ------  90 / 309 Accuracy: 35 / 160 = 0.21875 Loss: 2.682373046875
Train Batch ------  100 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 2.3068394660949707
Validation Accuracy: 126 / 500 = 0.252
Train Batch ------  110 / 309 Accuracy: 39 / 160 = 0.24375 Loss: 2.4709455966949463
Train Batch ------  120 / 309 Accuracy: 48 / 160 = 0.3 Loss: 2.6881191730499268
Train Batch ------  130 / 309 Accuracy: 36 / 160 = 0.225 Loss: 2.2133471965789795
Train Batch ------  140 / 309 Accuracy: 46 / 160 = 0.2875 Loss: 2.7854297161102295
Train Batch ------  150 / 309 Accuracy: 45 / 160 = 0.28125 Loss: 2.4222733974456787
Validation Accuracy: 121 / 500 = 0.242
Train Batch ------  160 / 309 Accuracy: 42 / 160 = 0.2625 Loss: 2.1281492710113525
Train Batch ------  170 / 309 Accuracy: 40 / 160 = 0.25 Loss: 2.0879111289978027
Train Batch ------  180 / 309 Accuracy: 31 / 160 = 0.19375 Loss: 2.11954927444458
Train Batch ------  190 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 1.8071644306182861
Train Batch ------  200 / 309 Accuracy: 47 / 160 = 0.29375 Loss: 2.4367544651031494
Validation Accuracy: 127 / 500 = 0.254
Train Batch ------  210 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 2.1438663005828857
Train Batch ------  220 / 309 Accuracy: 39 / 160 = 0.24375 Loss: 1.4150804281234741
Train Batch ------  230 / 309 Accuracy: 55 / 160 = 0.34375 Loss: 2.2470860481262207
Train Batch ------  240 / 309 Accuracy: 47 / 160 = 0.29375 Loss: 2.1401875019073486
Train Batch ------  250 / 309 Accuracy: 51 / 160 = 0.31875 Loss: 2.495779275894165
Validation Accuracy: 126 / 500 = 0.252
Train Batch ------  260 / 309 Accuracy: 38 / 160 = 0.2375 Loss: 2.22971248626709
Train Batch ------  270 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 2.1283953189849854
Train Batch ------  280 / 309 Accuracy: 31 / 160 = 0.19375 Loss: 1.784540057182312
Train Batch ------  290 / 309 Accuracy: 47 / 160 = 0.29375 Loss: 2.2764995098114014
Train Batch ------  300 / 309 Accuracy: 45 / 160 = 0.28125 Loss: 2.2312519550323486
Validation Accuracy: 129 / 500 = 0.258
Training Epoch 1
Train Batch ------  0 / 309 Accuracy: 3 / 16 = 0.1875 Loss: 1.740925669670105
Validation Accuracy: 118 / 500 = 0.236
Train Batch ------  10 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 1.7253071069717407
Train Batch ------  20 / 309 Accuracy: 33 / 160 = 0.20625 Loss: 1.5542153120040894
Train Batch ------  30 / 309 Accuracy: 49 / 160 = 0.30625 Loss: 1.6201324462890625
Train Batch ------  40 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 1.857405662536621
Train Batch ------  50 / 309 Accuracy: 48 / 160 = 0.3 Loss: 1.5567011833190918
Validation Accuracy: 129 / 500 = 0.258
Train Batch ------  60 / 309 Accuracy: 35 / 160 = 0.21875 Loss: 1.7200849056243896


# V2
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
learning_rate = 5e-5 # max learning rate
device_type = 'cpu'
epochs = 2
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

---

Training Epoch 0
Train Batch ------  0 / 309 Accuracy: 0 / 16 = 0.0 Loss: 5.667468547821045
Validation Accuracy: 4 / 500 = 0.008 Loss: 2.6431257650256157
Train Batch ------  10 / 309 Accuracy: 35 / 160 = 0.21875 Loss: 2.145542621612549
Train Batch ------  20 / 309 Accuracy: 33 / 160 = 0.20625 Loss: 2.0042784214019775
Train Batch ------  30 / 309 Accuracy: 33 / 160 = 0.20625 Loss: 2.1059024333953857
Train Batch ------  40 / 309 Accuracy: 36 / 160 = 0.225 Loss: 1.837001919746399
Train Batch ------  50 / 309 Accuracy: 30 / 160 = 0.1875 Loss: 2.1435060501098633
Validation Accuracy: 110 / 500 = 0.22 Loss: 1.8335949815809727
Train Batch ------  60 / 309 Accuracy: 45 / 160 = 0.28125 Loss: 1.6900817155838013
Train Batch ------  70 / 309 Accuracy: 37 / 160 = 0.23125 Loss: 1.8223611116409302
Train Batch ------  80 / 309 Accuracy: 43 / 160 = 0.26875 Loss: 1.892486572265625
Train Batch ------  90 / 309 Accuracy: 50 / 160 = 0.3125 Loss: 1.6812870502471924
Train Batch ------  100 / 309 Accuracy: 42 / 160 = 0.2625 Loss: 1.3245372772216797
Validation Accuracy: 109 / 500 = 0.218 Loss: 1.7852435447275639
Train Batch ------  110 / 309 Accuracy: 36 / 160 = 0.225 Loss: 2.1453075408935547
Train Batch ------  120 / 309 Accuracy: 48 / 160 = 0.3 Loss: 1.6629488468170166
Train Batch ------  130 / 309 Accuracy: 35 / 160 = 0.21875 Loss: 1.73581063747406
Train Batch ------  140 / 309 Accuracy: 36 / 160 = 0.225 Loss: 1.7181949615478516
Train Batch ------  150 / 309 Accuracy: 39 / 160 = 0.24375 Loss: 1.6473289728164673
Validation Accuracy: 110 / 500 = 0.22 Loss: 1.770024511963129
Train Batch ------  160 / 309 Accuracy: 39 / 160 = 0.24375 Loss: 1.832387924194336
Train Batch ------  170 / 309 Accuracy: 48 / 160 = 0.3 Loss: 1.9983010292053223
Train Batch ------  180 / 309 Accuracy: 40 / 160 = 0.25 Loss: 1.398314118385315
Train Batch ------  190 / 309 Accuracy: 31 / 160 = 0.19375 Loss: 1.8702890872955322
Train Batch ------  200 / 309 Accuracy: 44 / 160 = 0.275 Loss: 1.9285027980804443
Validation Accuracy: 126 / 500 = 0.252 Loss: 1.7536579705774784
Train Batch ------  210 / 309 Accuracy: 40 / 160 = 0.25 Loss: 2.0296144485473633
Train Batch ------  220 / 309 Accuracy: 49 / 160 = 0.30625 Loss: 1.6883115768432617
Train Batch ------  230 / 309 Accuracy: 32 / 160 = 0.2 Loss: 2.049046277999878
Train Batch ------  240 / 309 Accuracy: 37 / 160 = 0.23125 Loss: 1.3061120510101318
Train Batch ------  250 / 309 Accuracy: 42 / 160 = 0.2625 Loss: 1.3409677743911743
Validation Accuracy: 126 / 500 = 0.252 Loss: 1.736469142138958
Train Batch ------  260 / 309 Accuracy: 44 / 160 = 0.275 Loss: 1.6030097007751465
Train Batch ------  270 / 309 Accuracy: 42 / 160 = 0.2625 Loss: 1.8966014385223389
Train Batch ------  280 / 309 Accuracy: 44 / 160 = 0.275 Loss: 1.6497865915298462
Train Batch ------  290 / 309 Accuracy: 41 / 160 = 0.25625 Loss: 2.0818238258361816
Train Batch ------  300 / 309 Accuracy: 47 / 160 = 0.29375 Loss: 1.7418560981750488
Validation Accuracy: 145 / 500 = 0.29 Loss: 1.7179406322538853
Training Epoch 1
