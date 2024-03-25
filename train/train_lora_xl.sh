
export MODEL_NAME=/home/imgen/models/SDXL/juggernautXL_v8Rundiffusion/
export OUTPUT_DIR=`pwd`/digitalben-lora
export DATASET_NAME=`pwd`/new_ben_crop_small/
export DATASET_NAME=`pwd`/new_ben_crop_verysmall/

accelerate launch examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
	    --dataset_name=$DATASET_NAME \
	      --dataloader_num_workers=1 \
	        --resolution=1024 \
		  --center_crop \
		      --train_batch_size=1 \
		        --gradient_accumulation_steps=3 \
			  --max_train_steps=15000 \
			    --learning_rate=1e-04 \
			      --max_grad_norm=1 \
			        --lr_scheduler="cosine" \
				--num_validation_images=1 \
				--validation_epochs=40 \
				--validation_prompt="digitalben is giving a speech on a beach" \
				  --lr_warmup_steps=0 \
				    --output_dir=${OUTPUT_DIR} \
				    --mixed_precision='no' \
				    --gradient_checkpointing \
					    --checkpointing_steps=500 \
					        --seed=1337 \
						--random_flip
				    #--enable_xformers_memory_efficient_attention \
				    #--use_8bit_adam \
