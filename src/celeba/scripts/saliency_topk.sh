#python -W ignore src/celeba/topk_celeba.py --attack topk --evaluate_subset --image_save_dir /data/mariammaa/celeba/perturbed_test_topk_set2/

python -W ignore src/celeba/manipulate_eval_celeba.py --sal_method saliency_map --attack topk --image_save_dir /data/mariammaa/celeba/perturbed_test_topk_set2/ --tasks 0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39 --evaluate_subset --output_dir /data/mariammaa/celeba/results_topk_set2_saliencymap/





# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster2-0_3_4_7_10_12_13_14_15_16_17_20_22_26_28_30_35_38resnext-lr:0.01-wd:0.0_faithful-paper-20 --tasks 0_3_4_7_10_12_13_14_15_16_17_20_22_26_28_30_35_38
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster21_2_5_6_8_9_11_18_19_21_23_24_25_27_29_31_32_33_34_36_37_39resnext-lr:0.01-wd:0.0_visionary-star-13 --tasks 1_2_5_6_8_9_11_18_19_21_23_24_25_27_29_31_32_33_34_36_37_39

# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster30_3_7_8_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_radiant-paper-14 --tasks 0_3_7_8_12_16_20_22_30_38
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster31_2_5_11_18_19_21_24_25_27_31_32_33_34_36_39resnext-lr:0.01-wd:0.0_zesty-dragon-15 --tasks 1_2_5_11_18_19_21_24_25_27_31_32_33_34_36_39
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster34_6_9_10_13_14_15_17_23_26_28_29_35_37resnext-lr:0.01-wd:0.0_upbeat-feather-16 --tasks 4_6_9_10_13_14_15_17_23_26_28_29_35_37

# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster40_7_16_20_22_30_38resnext-lr:0.01-wd:0.0_unique-terrain-17 --tasks 0_7_16_20_22_30_38
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster4-10_13_14_15_17_26_29_37resnext-lr:0.01-wd:0.0_misty-dream-21 --tasks 10_13_14_15_17_26_29_37
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster41_3_4_5_6_9_11_23_24_27_28_32_33_35_39resnext-lr:0.01-wd:0.0_stilted-frog-18 --tasks 1_3_4_5_6_9_11_23_24_27_28_32_33_35_39
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster42_8_12_18_19_21_25_31_34_36resnext-lr:0.01-wd:0.0_eager-river-19 --tasks 2_8_12_18_19_21_25_31_34_36

# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster5-0_12_16_22_30_38resnext-lr:0.01-wd:0.0_decent-deluge-22 --tasks 0_12_16_22_30_38
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster5-1_2_5_6_9_11_26_27_32_37resnext-lr:0.01-wd:0.0_light-cosmos-23 --tasks 1_2_5_6_9_11_26_27_32_37
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster5-18_19_21_25_29_31_34_36resnext-lr:0.01-wd:0.0_exalted-serenity-26 --tasks 18_19_21_25_29_31_34_36
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster5-3_8_20_24_39resnext-lr:0.01-wd:0.0_eager-dew-24 --tasks 3_8_20_24_39
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster5-4_7_10_13_14_15_17_23_28_33_35resnext-lr:0.01-wd:0.0_vibrant-brook-25 --tasks 4_7_10_13_14_15_17_23_28_33_35

# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster7-0_12_16_20_22_30_38resnext-lr:0.01-wd:0.0_colorful-donkey-33 --tasks 0_12_16_20_22_30_38
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster7-1_6_18_27_34_36_37resnext-lr:0.01-wd:0.0_misunderstood-leaf-34 --tasks 1_6_18_27_34_36_37
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster7-19_21_25_29_31resnext-lr:0.01-wd:0.0_easy-wind-39 --tasks 19_21_25_29_31
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster7-2_3_8_24_39resnext-lr:0.01-wd:0.0_super-dawn-35 --tasks 2_3_8_24_39
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster7-4_5_28_33_35resnext-lr:0.01-wd:0.0_fanciful-salad-36 --tasks 4_5_28_33_35
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster7-7_10_13_14_15_23_32resnext-lr:0.01-wd:0.0_eager-snow-37 --tasks 7_10_13_14_15_23_32
# python -W ignore src/celeba/eval_celeba.py --net_basename _cluster7-9_11_17_26resnext-lr:0.01-wd:0.0_faithful-shadow-38 --tasks 9_11_17_26



