editing_prompt="let her wear batman eyemask"
editing_target='tex'
w_reg_diffuse=15000
edit_prompt_cfg=15
edit_img_cfg=2
w_tex=0.5
w_texYuv=1

python main.py --stage "edit" --text="$editing_prompt"  \
--edit_scope $editing_target --exp_root exp --exp_name demo --total_steps 201 --save_freq 50 \
--sds_input rendered --vis_att True --texture_generation latent --latent_sds_steps 0 --attention_reg_diffuse True --attention_sds True \
--w_reg_diffuse=$w_reg_diffuse --edit_prompt_cfg=$edit_prompt_cfg --edit_img_cfg=$edit_img_cfg --w_texSD=$w_tex --w_texYuv=$w_texYuv \
--load_id_path "./exp/demo/a zoomed out DSLR photo of Emma Watson/coarse geometry generation/seed42/200_coeff.npy" \
--load_diffuse_path "./exp/demo/a zoomed out DSLR photo of Emma Watson/texture generation/seed42/latent/400_diffuse_latent.npy"