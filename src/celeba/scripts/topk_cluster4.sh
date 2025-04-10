python -W ignore src/celeba/train_celeba_partial.py --tasks 0 8 12 16 20 22 30 38 --cluster_label topk_set2_cluster4 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 1 2 3 5 6 11 24 27 32 33 39 --cluster_label topk_set2_cluster4 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 4 7 9 10 13 14 17 23 26 28 35 37 --cluster_label topk_set2_cluster4 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 15 18 19 21 25 29 31 34 36 --cluster_label topk_set2_cluster4 --store_models

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-0_16_20_22_30_38resnext-lr:0.01-wd:0.0_lemon-thunder-45 --tasks 0_16_20_22_30_38 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-10_11_13_14_15_17_26_29_37resnext-lr:0.01-wd:0.0_neat-lake-48 --tasks 10_11_13_14_15_17_26_29_37 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-1_3_4_5_6_7_9_23_24_28_33_35resnext-lr:0.01-wd:0.0_colorful-dew-46 --tasks 1_3_4_5_6_7_9_23_24_28_33_35 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-2_8_12_18_19_21_25_27_31_32_34_36_39resnext-lr:0.01-wd:0.0_fanciful-pond-47 --tasks 2_8_12_18_19_21_25_27_31_32_34_36_39 --sal_method guidedBackprop --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-0_16_20_22_30_38resnext-lr:0.01-wd:0.0_lemon-thunder-45 --tasks 0_16_20_22_30_38 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-10_11_13_14_15_17_26_29_37resnext-lr:0.01-wd:0.0_neat-lake-48 --tasks 10_11_13_14_15_17_26_29_37 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-1_3_4_5_6_7_9_23_24_28_33_35resnext-lr:0.01-wd:0.0_colorful-dew-46 --tasks 1_3_4_5_6_7_9_23_24_28_33_35 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-2_8_12_18_19_21_25_27_31_32_34_36_39resnext-lr:0.01-wd:0.0_fanciful-pond-47 --tasks 2_8_12_18_19_21_25_27_31_32_34_36_39 --sal_method input_x_gradient --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-0_16_20_22_30_38resnext-lr:0.01-wd:0.0_lemon-thunder-45 --tasks 0_16_20_22_30_38 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-10_11_13_14_15_17_26_29_37resnext-lr:0.01-wd:0.0_neat-lake-48 --tasks 10_11_13_14_15_17_26_29_37 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-1_3_4_5_6_7_9_23_24_28_33_35resnext-lr:0.01-wd:0.0_colorful-dew-46 --tasks 1_3_4_5_6_7_9_23_24_28_33_35 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-2_8_12_18_19_21_25_27_31_32_34_36_39resnext-lr:0.01-wd:0.0_fanciful-pond-47 --tasks 2_8_12_18_19_21_25_27_31_32_34_36_39 --sal_method guided_gradcam --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-0_16_20_22_30_38resnext-lr:0.01-wd:0.0_lemon-thunder-45 --tasks 0_16_20_22_30_38 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-10_11_13_14_15_17_26_29_37resnext-lr:0.01-wd:0.0_neat-lake-48 --tasks 10_11_13_14_15_17_26_29_37 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-1_3_4_5_6_7_9_23_24_28_33_35resnext-lr:0.01-wd:0.0_colorful-dew-46 --tasks 1_3_4_5_6_7_9_23_24_28_33_35 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-2_8_12_18_19_21_25_27_31_32_34_36_39resnext-lr:0.01-wd:0.0_fanciful-pond-47 --tasks 2_8_12_18_19_21_25_27_31_32_34_36_39 --sal_method saliency_map

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-0_16_20_22_30_38resnext-lr:0.01-wd:0.0_lemon-thunder-45 --tasks 0_16_20_22_30_38 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-10_11_13_14_15_17_26_29_37resnext-lr:0.01-wd:0.0_neat-lake-48 --tasks 10_11_13_14_15_17_26_29_37 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-1_3_4_5_6_7_9_23_24_28_33_35resnext-lr:0.01-wd:0.0_colorful-dew-46 --tasks 1_3_4_5_6_7_9_23_24_28_33_35 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-2_8_12_18_19_21_25_27_31_32_34_36_39resnext-lr:0.01-wd:0.0_fanciful-pond-47 --tasks 2_8_12_18_19_21_25_27_31_32_34_36_39 --sal_method gradcam

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-0_16_20_22_30_38resnext-lr:0.01-wd:0.0_lemon-thunder-45 --tasks 0_16_20_22_30_38 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-10_11_13_14_15_17_26_29_37resnext-lr:0.01-wd:0.0_neat-lake-48 --tasks 10_11_13_14_15_17_26_29_37 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-1_3_4_5_6_7_9_23_24_28_33_35resnext-lr:0.01-wd:0.0_colorful-dew-46 --tasks 1_3_4_5_6_7_9_23_24_28_33_35 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster4-2_8_12_18_19_21_25_27_31_32_34_36_39resnext-lr:0.01-wd:0.0_fanciful-pond-47 --tasks 2_8_12_18_19_21_25_27_31_32_34_36_39 --sal_method integrated_gradients
