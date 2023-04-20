echo "resnet_v2"
python atta.py  --source_model resnet_v2 --output_dir ./outputs/atta_resnet_v2_final4 --cuda_device 1 &&
python simple_eval.py --EXP_ID atta --input_dir ./outputs/atta_resnet_v2_final4/adv-debug --source_model resnet_v2 &&
echo "inceptionv3"
python atta.py  --source_model InceptionV3 --output_dir ./outputs/atta_inceptionv3_final4 --cuda_device 2
python simple_eval.py --EXP_ID atta --input_dir ./outputs/atta_inceptionv3_final4/adv-debug --source_model InceptionV3 &&
echo "InceptionV4"
python atta.py  --source_model InceptionV4 --output_dir ./outputs/atta_inceptionv4_final4 --cuda_device 2 &&
python simple_eval.py --EXP_ID atta --input_dir ./outputs/atta_inceptionv4_final4/adv-debug --source_model InceptionV4 &&
echo "InceptionResnetV2"
python atta.py  --source_model InceptionResnetV2 --output_dir ./outputs/atta_InceptionResnetV2_final4  --cuda_device 3 &&
python simple_eval.py --EXP_ID atta --input_dir ./outputs/atta_InceptionResnetV2_final4/adv-debug --source_model InceptionResnetV2 --cuda_device 1