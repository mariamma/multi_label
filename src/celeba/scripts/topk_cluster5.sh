python -W ignore src/celeba/train_celeba_partial.py --tasks 0 12 16 20 22 30 38 --cluster_label topk_set2_cluster5 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 1 18 26 29 34 36 37 --cluster_label topk_set2_cluster5 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 2 3 5 6 11 24 27 32 33 39 --cluster_label topk_set2_cluster5 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 4 7 9 10 13 14 17 23 28 35 --cluster_label topk_set2_cluster5 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 8 15 19 21 25 31 --cluster_label topk_set2_cluster5 --store_models

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-0_12_16_20_22_30_32_38resnext-lr:0.01-wd:0.0_neat-sun-49 --tasks 0_12_16_20_22_30_32_38 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-3_4_6_7_13_14_17_23_28_33_35resnext-lr:0.01-wd:0.0_scarlet-snow-52 --tasks 3_4_6_7_13_14_17_23_28_33_35 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_golden-wind-53 --tasks 8_15_19_21_25_31 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-2_5_24_39resnext-lr:0.01-wd:0.0_grateful-cherry-51 --tasks 2_5_24_39 --sal_method guidedBackprop --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-0_12_16_20_22_30_32_38resnext-lr:0.01-wd:0.0_neat-sun-49 --tasks 0_12_16_20_22_30_32_38 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-3_4_6_7_13_14_17_23_28_33_35resnext-lr:0.01-wd:0.0_scarlet-snow-52 --tasks 3_4_6_7_13_14_17_23_28_33_35 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_golden-wind-53 --tasks 8_15_19_21_25_31 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-2_5_24_39resnext-lr:0.01-wd:0.0_grateful-cherry-51 --tasks 2_5_24_39 --sal_method input_x_gradient --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-0_12_16_20_22_30_32_38resnext-lr:0.01-wd:0.0_neat-sun-49 --tasks 0_12_16_20_22_30_32_38 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-3_4_6_7_13_14_17_23_28_33_35resnext-lr:0.01-wd:0.0_scarlet-snow-52 --tasks 3_4_6_7_13_14_17_23_28_33_35 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_golden-wind-53 --tasks 8_15_19_21_25_31 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-2_5_24_39resnext-lr:0.01-wd:0.0_grateful-cherry-51 --tasks 2_5_24_39 --sal_method guided_gradcam --evaluate_subset


# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-0_12_16_20_22_30_32_38resnext-lr:0.01-wd:0.0_neat-sun-49 --tasks 0_12_16_20_22_30_32_38 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-3_4_6_7_13_14_17_23_28_33_35resnext-lr:0.01-wd:0.0_scarlet-snow-52 --tasks 3_4_6_7_13_14_17_23_28_33_35 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_golden-wind-53 --tasks 8_15_19_21_25_31 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-2_5_24_39resnext-lr:0.01-wd:0.0_grateful-cherry-51 --tasks 2_5_24_39 --sal_method saliency_map

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-0_12_16_20_22_30_32_38resnext-lr:0.01-wd:0.0_neat-sun-49 --tasks 0_12_16_20_22_30_32_38 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-3_4_6_7_13_14_17_23_28_33_35resnext-lr:0.01-wd:0.0_scarlet-snow-52 --tasks 3_4_6_7_13_14_17_23_28_33_35 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_golden-wind-53 --tasks 8_15_19_21_25_31 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-2_5_24_39resnext-lr:0.01-wd:0.0_grateful-cherry-51 --tasks 2_5_24_39 --sal_method gradcam

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-0_12_16_20_22_30_32_38resnext-lr:0.01-wd:0.0_neat-sun-49 --tasks 0_12_16_20_22_30_32_38 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-3_4_6_7_13_14_17_23_28_33_35resnext-lr:0.01-wd:0.0_scarlet-snow-52 --tasks 3_4_6_7_13_14_17_23_28_33_35 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_golden-wind-53 --tasks 8_15_19_21_25_31 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster5-2_5_24_39resnext-lr:0.01-wd:0.0_grateful-cherry-51 --tasks 2_5_24_39 --sal_method integrated_gradients
