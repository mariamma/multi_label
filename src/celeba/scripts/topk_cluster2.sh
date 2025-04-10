python -W ignore src/celeba/train_celeba_partial.py --tasks 0 3 4 7 10 12 13 14 15 16 17 20 22 28 30 35 38 --cluster_label topk_set2_cluster2 --store_models
python -W ignore src/celeba/train_celeba_partial.py --tasks 1 2 5 6 8 9 11 18 19 21 23 24 25 26 27 29 31 32 33 34 36 37 39 --cluster_label topk_set2_cluster2 --store_models

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38resnext-lr:0.01-wd:0.0_silver-glade-40 --tasks 0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38 --sal_method guidedBackprop --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39resnext-lr:0.01-wd:0.0_lilac-elevator-41 --tasks 1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39 --sal_method guidedBackprop --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38resnext-lr:0.01-wd:0.0_silver-glade-40 --tasks 0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38 --sal_method input_x_gradient --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39resnext-lr:0.01-wd:0.0_lilac-elevator-41 --tasks 1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39 --sal_method input_x_gradient --evaluate_subset

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38resnext-lr:0.01-wd:0.0_silver-glade-40 --tasks 0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38 --sal_method guided_gradcam --evaluate_subset
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39resnext-lr:0.01-wd:0.0_lilac-elevator-41 --tasks 1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39 --sal_method guided_gradcam --evaluate_subset



# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38resnext-lr:0.01-wd:0.0_silver-glade-40 --tasks 0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38 --sal_method saliency_map
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39resnext-lr:0.01-wd:0.0_lilac-elevator-41 --tasks 1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39 --sal_method saliency_map

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38resnext-lr:0.01-wd:0.0_silver-glade-40 --tasks 0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38 --sal_method gradcam
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39resnext-lr:0.01-wd:0.0_lilac-elevator-41 --tasks 1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39 --sal_method gradcam

# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38resnext-lr:0.01-wd:0.0_silver-glade-40 --tasks 0_3_4_6_7_9_10_12_13_14_15_16_17_22_23_26_28_29_30_35_37_38 --sal_method integrated_gradients
# python -W ignore src/celeba/multi_eval_celeba.py --net_basename _cluster2-1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39resnext-lr:0.01-wd:0.0_lilac-elevator-41 --tasks 1_2_5_8_11_18_19_20_21_24_25_27_31_32_33_34_36_39 --sal_method integrated_gradients

