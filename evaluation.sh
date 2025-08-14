INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_0/agent_0/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_0/agent_0/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_1/agent_0/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_1/agent_0/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_2/agent_0/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_2/agent_0/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_0/agent_1/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_0/agent_1/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_1/agent_1/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_1/agent_1/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_2/agent_1/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_2/agent_1/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_0/agent_2/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_0/agent_2/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_1/agent_2/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_1/agent_2/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d




INPUT_MESH=output/Replica/room1/Ours_With_NARUTO_2/agent_2/mesh_track665.ply
VIRT_CAM_PATH=eval_data/Replica/room1/virtual_cameras
python cull_mesh.py --config configs/Replica/room1.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose


REC_MESH=output/Replica/room1/Ours_With_NARUTO_2/agent_2/mesh_track665_cull_virt_cams.ply
GT_MESH=eval_data/Replica/room1/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -3d