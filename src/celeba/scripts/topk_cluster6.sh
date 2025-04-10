python -W ignore src/celeba/train_celeba_partial.py --tasks 0 12 16 22 30 38 --cluster_label topk_set2_cluster6 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 1 18 34 36 37 --cluster_label topk_set2_cluster6 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 2 8 20 24 27 32 39 --cluster_label topk_set2_cluster6 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 3 6 7 10 13 14 23 28 --cluster_label topk_set2_cluster6 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 4 5 9 11 17 26 33 35 --cluster_label topk_set2_cluster6 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 15 19 21 25 29 31 --cluster_label topk_set2_cluster6 --store_models

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-0_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_fanciful-night-54 --tasks 0_12_16_20_22_30_38 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-1_6_18_23_34_36_37resnext-lr:0.01-wd:0.0_astral-morning-55 --tasks 1_6_18_23_34_36_37 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-3_4_7_13_14_28_33_35resnext-lr:0.01-wd:0.0_daily-fire-57 --tasks 3_4_7_13_14_28_33_35 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_colorful-sky-58 --tasks 8_15_19_21_25_31 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-9_10_11_17_26_29resnext-lr:0.01-wd:0.0_silver-gorge-59 --tasks 9_10_11_17_26_29 --sal_method guidedBackprop --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-0_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_fanciful-night-54 --tasks 0_12_16_20_22_30_38 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-1_6_18_23_34_36_37resnext-lr:0.01-wd:0.0_astral-morning-55 --tasks 1_6_18_23_34_36_37 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-3_4_7_13_14_28_33_35resnext-lr:0.01-wd:0.0_daily-fire-57 --tasks 3_4_7_13_14_28_33_35 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_colorful-sky-58 --tasks 8_15_19_21_25_31 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-9_10_11_17_26_29resnext-lr:0.01-wd:0.0_silver-gorge-59 --tasks 9_10_11_17_26_29 --sal_method input_x_gradient --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-0_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_fanciful-night-54 --tasks 0_12_16_20_22_30_38 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-1_6_18_23_34_36_37resnext-lr:0.01-wd:0.0_astral-morning-55 --tasks 1_6_18_23_34_36_37 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-3_4_7_13_14_28_33_35resnext-lr:0.01-wd:0.0_daily-fire-57 --tasks 3_4_7_13_14_28_33_35 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_colorful-sky-58 --tasks 8_15_19_21_25_31 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-9_10_11_17_26_29resnext-lr:0.01-wd:0.0_silver-gorge-59 --tasks 9_10_11_17_26_29 --sal_method guided_gradcam --evaluate_subset



# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-0_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_fanciful-night-54 --tasks 0_12_16_20_22_30_38 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-1_6_18_23_34_36_37resnext-lr:0.01-wd:0.0_astral-morning-55 --tasks 1_6_18_23_34_36_37 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-3_4_7_13_14_28_33_35resnext-lr:0.01-wd:0.0_daily-fire-57 --tasks 3_4_7_13_14_28_33_35 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_colorful-sky-58 --tasks 8_15_19_21_25_31 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-9_10_11_17_26_29resnext-lr:0.01-wd:0.0_silver-gorge-59 --tasks 9_10_11_17_26_29 --sal_method saliency_map

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-0_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_fanciful-night-54 --tasks 0_12_16_20_22_30_38 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-1_6_18_23_34_36_37resnext-lr:0.01-wd:0.0_astral-morning-55 --tasks 1_6_18_23_34_36_37 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-3_4_7_13_14_28_33_35resnext-lr:0.01-wd:0.0_daily-fire-57 --tasks 3_4_7_13_14_28_33_35 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_colorful-sky-58 --tasks 8_15_19_21_25_31 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-9_10_11_17_26_29resnext-lr:0.01-wd:0.0_silver-gorge-59 --tasks 9_10_11_17_26_29 --sal_method gradcam

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-0_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_fanciful-night-54 --tasks 0_12_16_20_22_30_38 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-1_6_18_23_34_36_37resnext-lr:0.01-wd:0.0_astral-morning-55 --tasks 1_6_18_23_34_36_37 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-3_4_7_13_14_28_33_35resnext-lr:0.01-wd:0.0_daily-fire-57 --tasks 3_4_7_13_14_28_33_35 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-8_15_19_21_25_31resnext-lr:0.01-wd:0.0_colorful-sky-58 --tasks 8_15_19_21_25_31 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster6-9_10_11_17_26_29resnext-lr:0.01-wd:0.0_silver-gorge-59 --tasks 9_10_11_17_26_29 --sal_method integrated_gradients


