echo "resnet_v2"
python atta_ni_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/atta_resnet_v2_final3 --cuda_device 1 --num_iter 10&&
python simple_eval.py --EXP_ID train_debug --input_dir ./outputs/atta_resnet_v2_final3/adv-debug --source_model resnet_v2 &&
echo "inceptionv3"
python atta_ni_fgsm.py  --source_model InceptionV3 --output_dir ./outputs/atta_inceptionv3_final &&
python atta_ni_fgsm.py  --source_model InceptionV3 --output_dir ./outputs/atta_inceptionv3_final2 --num_iter 10
python atta_ni_fgsm.py  --source_model InceptionV3 --output_dir ./outputs/atta_inceptionv3_final3 --num_iter 8 --alpha1 0.8441259307379171 --alpha2 0.6989752957059422 --gamma 0.03445142957980907 --num_epoch 13
python simple_eval.py --EXP_ID train_debug --input_dir ./outputs/atta_inceptionv3_final/adv-debug --source_model InceptionV3 &&
echo "InceptionV4"
python atta_ni_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/atta_inceptionv4_final3 --cuda_device 2 --num_iter 10&&
python simple_eval.py --EXP_ID train_debug --input_dir ./outputs/atta_inceptionv4_final3/adv-debug --source_model InceptionV4 &&
echo "InceptionResnetV2"
python atta_ni_fgsm_v2.py  --source_model InceptionResnetV2 --output_dir ./outputs/atta_InceptionResnetV2_final3  --cuda_device 3 &&
python simple_eval.py --EXP_ID train_debug --input_dir ./outputs/atta_InceptionResnetV2_final3/adv-debug --source_model InceptionResnetV2 