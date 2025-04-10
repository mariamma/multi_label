python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_saliencymap/ --evaluate_subset --sal_method saliency_map --net_basename full_cluster4-0_2_3_8-resnext-lr:0.01-wd:0.0_cool-aardvark-35 --tasks 0_2_3_8
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_saliencymap/ --evaluate_subset --sal_method saliency_map --net_basename full_cluster4-1_6_7_9_10_11_13-resnext-lr:0.01-wd:0.0_curious-galaxy-36 --tasks 1_6_7_9_10_11_13
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_saliencymap/ --evaluate_subset --sal_method saliency_map --net_basename full_cluster4-4_5_12-resnext-lr:0.01-wd:0.0_swept-jazz-37 --tasks 4_5_12

python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_gradcam/ --evaluate_subset --sal_method gradcam --net_basename full_cluster4-0_2_3_8-resnext-lr:0.01-wd:0.0_cool-aardvark-35 --tasks 0_2_3_8
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_gradcam/ --evaluate_subset --sal_method gradcam --net_basename full_cluster4-1_6_7_9_10_11_13-resnext-lr:0.01-wd:0.0_curious-galaxy-36 --tasks 1_6_7_9_10_11_13
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_gradcam/ --evaluate_subset --sal_method gradcam --net_basename full_cluster4-4_5_12-resnext-lr:0.01-wd:0.0_swept-jazz-37 --tasks 4_5_12

python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_inputxgradient/ --evaluate_subset --sal_method input_x_gradient --net_basename full_cluster4-0_2_3_8-resnext-lr:0.01-wd:0.0_cool-aardvark-35 --tasks 0_2_3_8
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_inputxgradient/ --evaluate_subset --sal_method input_x_gradient --net_basename full_cluster4-1_6_7_9_10_11_13-resnext-lr:0.01-wd:0.0_curious-galaxy-36 --tasks 1_6_7_9_10_11_13
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_inputxgradient/ --evaluate_subset --sal_method input_x_gradient --net_basename full_cluster4-4_5_12-resnext-lr:0.01-wd:0.0_swept-jazz-37 --tasks 4_5_12

python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_guidedbackprop/ --evaluate_subset --sal_method guidedBackprop --net_basename full_cluster4-0_2_3_8-resnext-lr:0.01-wd:0.0_cool-aardvark-35 --tasks 0_2_3_8
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_guidedbackprop/ --evaluate_subset --sal_method guidedBackprop --net_basename full_cluster4-1_6_7_9_10_11_13-resnext-lr:0.01-wd:0.0_curious-galaxy-36 --tasks 1_6_7_9_10_11_13
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_guidedbackprop/ --evaluate_subset --sal_method guidedBackprop --net_basename full_cluster4-4_5_12-resnext-lr:0.01-wd:0.0_swept-jazz-37 --tasks 4_5_12

python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_guidedgradcam/ --evaluate_subset --sal_method guided_gradcam --net_basename full_cluster4-0_2_3_8-resnext-lr:0.01-wd:0.0_cool-aardvark-35 --tasks 0_2_3_8
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_guidedgradcam/ --evaluate_subset --sal_method guided_gradcam --net_basename full_cluster4-1_6_7_9_10_11_13-resnext-lr:0.01-wd:0.0_curious-galaxy-36 --tasks 1_6_7_9_10_11_13
python -W ignore src/nih/multi_eval_nih.py --image_save_dir /data/mariammaa/nih_multi_label/perturbed_nih_topk/ --results_dir /data/mariammaa/nih_multi_label/nih_results_topk_guidedgradcam/ --evaluate_subset --sal_method guided_gradcam --net_basename full_cluster4-4_5_12-resnext-lr:0.01-wd:0.0_swept-jazz-37 --tasks 4_5_12

# python -W ignore train_nih_partial.py --tasks 0 2 3 8 --cluster_label cluster4 --model_storage /data/mariammaa/nih_multi_label/checkpoint_nih_topk/ --store_models

# python -W ignore train_nih_partial.py --tasks 1 6 7 9 10 11 13 --cluster_label cluster4 --model_storage /data/mariammaa/nih_multi_label/checkpoint_nih_topk/ --store_models

# python -W ignore train_nih_partial.py --tasks 4 5 12 --cluster_label cluster4 --model_storage /data/mariammaa/nih_multi_label/checkpoint_nih_topk/ --store_models