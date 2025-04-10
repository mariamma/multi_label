python -W ignore src/celeba/train_celeba_partial.py --tasks 0 3 7 12 15 16 20 22 30 38 --cluster_label topk_set2_cluster3 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 1 4 5 6 9 10 13 14 17 23 26 28 29 33 34 35 37 --cluster_label topk_set2_cluster3 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 2 8 11 18 19 21 24 25 27 31 32 36 39 --cluster_label topk_set2_cluster3 --store_models

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-0_3_7_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_super-aardvark-42 --tasks 0_3_7_12_16_20_22_30_38 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-1_2_5_8_18_19_21_24_25_27_31_32_33_36_39resnext-lr:0.01-wd:0.0_dutiful-cherry-43 --tasks 1_2_5_8_18_19_21_24_25_27_31_32_33_36_39 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37resnext-lr:0.01-wd:0.0_dandy-glade-44 --tasks 4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37 --sal_method guidedBackprop --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-0_3_7_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_super-aardvark-42 --tasks 0_3_7_12_16_20_22_30_38 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-1_2_5_8_18_19_21_24_25_27_31_32_33_36_39resnext-lr:0.01-wd:0.0_dutiful-cherry-43 --tasks 1_2_5_8_18_19_21_24_25_27_31_32_33_36_39 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37resnext-lr:0.01-wd:0.0_dandy-glade-44 --tasks 4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37 --sal_method input_x_gradient --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-0_3_7_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_super-aardvark-42 --tasks 0_3_7_12_16_20_22_30_38 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-1_2_5_8_18_19_21_24_25_27_31_32_33_36_39resnext-lr:0.01-wd:0.0_dutiful-cherry-43 --tasks 1_2_5_8_18_19_21_24_25_27_31_32_33_36_39 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37resnext-lr:0.01-wd:0.0_dandy-glade-44 --tasks 4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37 --sal_method guided_gradcam --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-0_3_7_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_super-aardvark-42 --tasks 0_3_7_12_16_20_22_30_38 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-1_2_5_8_18_19_21_24_25_27_31_32_33_36_39resnext-lr:0.01-wd:0.0_dutiful-cherry-43 --tasks 1_2_5_8_18_19_21_24_25_27_31_32_33_36_39 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37resnext-lr:0.01-wd:0.0_dandy-glade-44 --tasks 4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37 --sal_method saliency_map

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-0_3_7_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_super-aardvark-42 --tasks 0_3_7_12_16_20_22_30_38 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-1_2_5_8_18_19_21_24_25_27_31_32_33_36_39resnext-lr:0.01-wd:0.0_dutiful-cherry-43 --tasks 1_2_5_8_18_19_21_24_25_27_31_32_33_36_39 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37resnext-lr:0.01-wd:0.0_dandy-glade-44 --tasks 4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37 --sal_method gradcam

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-0_3_7_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_super-aardvark-42 --tasks 0_3_7_12_16_20_22_30_38 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-1_2_5_8_18_19_21_24_25_27_31_32_33_36_39resnext-lr:0.01-wd:0.0_dutiful-cherry-43 --tasks 1_2_5_8_18_19_21_24_25_27_31_32_33_36_39 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster3-4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37resnext-lr:0.01-wd:0.0_dandy-glade-44 --tasks 4_6_9_10_11_13_14_15_17_23_26_28_29_34_35_37 --sal_method integrated_gradients
