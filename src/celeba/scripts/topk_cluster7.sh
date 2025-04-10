python -W ignore src/celeba/train_celeba_partial.py --tasks 1 18 29 34 36 37 --cluster_label topk_set2_cluster7 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 2 3 6 8 20 24 27 39 --cluster_label topk_set2_cluster7 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 4 5 9 11 17 26 35 --cluster_label topk_set2_cluster7 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 7 10 13 14 23 28 --cluster_label topk_set2_cluster7 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 15 19 21 25 31 33 --cluster_label topk_set2_cluster7 --store_models

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-0_12_16_22_30_38resnext-lr:0.01-wd:0.0_major-disco-60 --tasks 0_12_16_22_30_38 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-3_7_13_14_23resnext-lr:0.01-wd:0.0_whole-resonance-63 --tasks 3_7_13_14_23 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-10_11_26_27_32resnext-lr:0.01-wd:0.0_zesty-spaceship-65 --tasks 10_11_26_27_32 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-15_19_21_25_29_31resnext-lr:0.01-wd:0.0_prime-bird-66 --tasks 15_19_21_25_29_31 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-1_6_18_34_36_37resnext-lr:0.01-wd:0.0_dark-salad-61 --tasks 1_6_18_34_36_37 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-4_5_9_17_28_33_35resnext-lr:0.01-wd:0.0_lyric-bee-64 --tasks 4_5_9_17_28_33_35 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-2_8_20_24_39resnext-lr:0.01-wd:0.0_likely-energy-62 --tasks 2_8_20_24_39 --sal_method guidedBackprop --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-0_12_16_22_30_38resnext-lr:0.01-wd:0.0_major-disco-60 --tasks 0_12_16_22_30_38 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-3_7_13_14_23resnext-lr:0.01-wd:0.0_whole-resonance-63 --tasks 3_7_13_14_23 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-10_11_26_27_32resnext-lr:0.01-wd:0.0_zesty-spaceship-65 --tasks 10_11_26_27_32 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-15_19_21_25_29_31resnext-lr:0.01-wd:0.0_prime-bird-66 --tasks 15_19_21_25_29_31 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-1_6_18_34_36_37resnext-lr:0.01-wd:0.0_dark-salad-61 --tasks 1_6_18_34_36_37 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-4_5_9_17_28_33_35resnext-lr:0.01-wd:0.0_lyric-bee-64 --tasks 4_5_9_17_28_33_35 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-2_8_20_24_39resnext-lr:0.01-wd:0.0_likely-energy-62 --tasks 2_8_20_24_39 --sal_method input_x_gradient --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-0_12_16_22_30_38resnext-lr:0.01-wd:0.0_major-disco-60 --tasks 0_12_16_22_30_38 --sal_method guided_gradcam --evaluate_subset 
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-3_7_13_14_23resnext-lr:0.01-wd:0.0_whole-resonance-63 --tasks 3_7_13_14_23 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-10_11_26_27_32resnext-lr:0.01-wd:0.0_zesty-spaceship-65 --tasks 10_11_26_27_32 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-15_19_21_25_29_31resnext-lr:0.01-wd:0.0_prime-bird-66 --tasks 15_19_21_25_29_31 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-1_6_18_34_36_37resnext-lr:0.01-wd:0.0_dark-salad-61 --tasks 1_6_18_34_36_37 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-4_5_9_17_28_33_35resnext-lr:0.01-wd:0.0_lyric-bee-64 --tasks 4_5_9_17_28_33_35 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-2_8_20_24_39resnext-lr:0.01-wd:0.0_likely-energy-62 --tasks 2_8_20_24_39 --sal_method guided_gradcam --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-0_12_16_22_30_38resnext-lr:0.01-wd:0.0_major-disco-60 --tasks 0_12_16_22_30_38 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-3_7_13_14_23resnext-lr:0.01-wd:0.0_whole-resonance-63 --tasks 3_7_13_14_23 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-10_11_26_27_32resnext-lr:0.01-wd:0.0_zesty-spaceship-65 --tasks 10_11_26_27_32 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-15_19_21_25_29_31resnext-lr:0.01-wd:0.0_prime-bird-66 --tasks 15_19_21_25_29_31 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-1_6_18_34_36_37resnext-lr:0.01-wd:0.0_dark-salad-61 --tasks 1_6_18_34_36_37 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-4_5_9_17_28_33_35resnext-lr:0.01-wd:0.0_lyric-bee-64 --tasks 4_5_9_17_28_33_35 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-2_8_20_24_39resnext-lr:0.01-wd:0.0_likely-energy-62 --tasks 2_8_20_24_39 --sal_method saliency_map

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-0_12_16_22_30_38resnext-lr:0.01-wd:0.0_major-disco-60 --tasks 0_12_16_22_30_38 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-3_7_13_14_23resnext-lr:0.01-wd:0.0_whole-resonance-63 --tasks 3_7_13_14_23 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-10_11_26_27_32resnext-lr:0.01-wd:0.0_zesty-spaceship-65 --tasks 10_11_26_27_32 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-15_19_21_25_29_31resnext-lr:0.01-wd:0.0_prime-bird-66 --tasks 15_19_21_25_29_31 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-1_6_18_34_36_37resnext-lr:0.01-wd:0.0_dark-salad-61 --tasks 1_6_18_34_36_37 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-4_5_9_17_28_33_35resnext-lr:0.01-wd:0.0_lyric-bee-64 --tasks 4_5_9_17_28_33_35 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-2_8_20_24_39resnext-lr:0.01-wd:0.0_likely-energy-62 --tasks 2_8_20_24_39 --sal_method gradcam

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-0_12_16_22_30_38resnext-lr:0.01-wd:0.0_major-disco-60 --tasks 0_12_16_22_30_38 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-3_7_13_14_23resnext-lr:0.01-wd:0.0_whole-resonance-63 --tasks 3_7_13_14_23 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-10_11_26_27_32resnext-lr:0.01-wd:0.0_zesty-spaceship-65 --tasks 10_11_26_27_32 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-15_19_21_25_29_31resnext-lr:0.01-wd:0.0_prime-bird-66 --tasks 15_19_21_25_29_31 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-1_6_18_34_36_37resnext-lr:0.01-wd:0.0_dark-salad-61 --tasks 1_6_18_34_36_37 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-4_5_9_17_28_33_35resnext-lr:0.01-wd:0.0_lyric-bee-64 --tasks 4_5_9_17_28_33_35 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster7-2_8_20_24_39resnext-lr:0.01-wd:0.0_likely-energy-62 --tasks 2_8_20_24_39 --sal_method integrated_gradients