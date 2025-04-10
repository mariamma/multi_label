python -W ignore src/celeba/train_celeba_partial.py --tasks 2 8 24 27 32 39 --cluster_label topk_set2_cluster8 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 3 4 7 28 --cluster_label topk_set2_cluster8 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 5 6 33 --cluster_label topk_set2_cluster8 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 9 11 17 26 35 --cluster_label topk_set2_cluster8 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 10 13 14 15 23 --cluster_label topk_set2_cluster8 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 19 21 25 31 --cluster_label topk_set2_cluster8 --store_models

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-0_16_22_30_38resnext-lr:0.01-wd:0.0_driven-grass-67 --tasks 0_16_22_30_38 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-3_7_13_14resnext-lr:0.01-wd:0.0_young-dawn-70 --tasks 3_7_13_14 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-1_18_34_36resnext-lr:0.01-wd:0.0_distinctive-lion-68 --tasks 1_18_34_36 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-6_10_11_12_23_26_27_29_37resnext-lr:0.01-wd:0.0_azure-bee-72 --tasks 6_10_11_12_23_26_27_29_37 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-15_19_21_25_31_33resnext-lr:0.01-wd:0.0_lucky-eon-73 --tasks 15_19_21_25_31_33 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-2_8_20_24_39resnext-lr:0.01-wd:0.0_copper-frost-69 --tasks 2_8_20_24_39 --sal_method guidedBackprop --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-0_16_22_30_38resnext-lr:0.01-wd:0.0_driven-grass-67 --tasks 0_16_22_30_38 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-3_7_13_14resnext-lr:0.01-wd:0.0_young-dawn-70 --tasks 3_7_13_14 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-1_18_34_36resnext-lr:0.01-wd:0.0_distinctive-lion-68 --tasks 1_18_34_36 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-6_10_11_12_23_26_27_29_37resnext-lr:0.01-wd:0.0_azure-bee-72 --tasks 6_10_11_12_23_26_27_29_37 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-15_19_21_25_31_33resnext-lr:0.01-wd:0.0_lucky-eon-73 --tasks 15_19_21_25_31_33 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-2_8_20_24_39resnext-lr:0.01-wd:0.0_copper-frost-69 --tasks 2_8_20_24_39 --sal_method input_x_gradient --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-0_16_22_30_38resnext-lr:0.01-wd:0.0_driven-grass-67 --tasks 0_16_22_30_38 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-3_7_13_14resnext-lr:0.01-wd:0.0_young-dawn-70 --tasks 3_7_13_14 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-1_18_34_36resnext-lr:0.01-wd:0.0_distinctive-lion-68 --tasks 1_18_34_36 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-6_10_11_12_23_26_27_29_37resnext-lr:0.01-wd:0.0_azure-bee-72 --tasks 6_10_11_12_23_26_27_29_37 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-15_19_21_25_31_33resnext-lr:0.01-wd:0.0_lucky-eon-73 --tasks 15_19_21_25_31_33 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-2_8_20_24_39resnext-lr:0.01-wd:0.0_copper-frost-69 --tasks 2_8_20_24_39 --sal_method guided_gradcam --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-0_16_22_30_38resnext-lr:0.01-wd:0.0_driven-grass-67 --tasks 0_16_22_30_38 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-3_7_13_14resnext-lr:0.01-wd:0.0_young-dawn-70 --tasks 3_7_13_14 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-1_18_34_36resnext-lr:0.01-wd:0.0_distinctive-lion-68 --tasks 1_18_34_36 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-6_10_11_12_23_26_27_29_37resnext-lr:0.01-wd:0.0_azure-bee-72 --tasks 6_10_11_12_23_26_27_29_37 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-15_19_21_25_31_33resnext-lr:0.01-wd:0.0_lucky-eon-73 --tasks 15_19_21_25_31_33 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-2_8_20_24_39resnext-lr:0.01-wd:0.0_copper-frost-69 --tasks 2_8_20_24_39 --sal_method saliency_map

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-0_16_22_30_38resnext-lr:0.01-wd:0.0_driven-grass-67 --tasks 0_16_22_30_38 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-3_7_13_14resnext-lr:0.01-wd:0.0_young-dawn-70 --tasks 3_7_13_14 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-1_18_34_36resnext-lr:0.01-wd:0.0_distinctive-lion-68 --tasks 1_18_34_36 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-6_10_11_12_23_26_27_29_37resnext-lr:0.01-wd:0.0_azure-bee-72 --tasks 6_10_11_12_23_26_27_29_37 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-15_19_21_25_31_33resnext-lr:0.01-wd:0.0_lucky-eon-73 --tasks 15_19_21_25_31_33 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-2_8_20_24_39resnext-lr:0.01-wd:0.0_copper-frost-69 --tasks 2_8_20_24_39 --sal_method gradcam

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-0_16_22_30_38resnext-lr:0.01-wd:0.0_driven-grass-67 --tasks 0_16_22_30_38 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-3_7_13_14resnext-lr:0.01-wd:0.0_young-dawn-70 --tasks 3_7_13_14 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-1_18_34_36resnext-lr:0.01-wd:0.0_distinctive-lion-68 --tasks 1_18_34_36 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-6_10_11_12_23_26_27_29_37resnext-lr:0.01-wd:0.0_azure-bee-72 --tasks 6_10_11_12_23_26_27_29_37 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-15_19_21_25_31_33resnext-lr:0.01-wd:0.0_lucky-eon-73 --tasks 15_19_21_25_31_33 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster8-2_8_20_24_39resnext-lr:0.01-wd:0.0_copper-frost-69 --tasks 2_8_20_24_39 --sal_method integrated_gradients
