
                            #!/usr/bin/bash

                            export CUDA_VISIBLE_DEVICES=0

                            model_path=/
                            data_filepath=/
                            save_filepath=/
                            prompt_column_name=a
                            tensor_parallel_size=1

                            python src/autoalign/data/instruction/back_translation.py \
                                --reverse \
                                --model_path=/ \
                                --data_filepath=/ \
                                --save_filepath=/ \
                                --prompt_column_name=a \
                                --tensor_parallel_size=1
                            