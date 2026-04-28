import bpy
import os
import pandas as pd

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_render():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.device = 'GPU'

def setup_camera():
    bpy.ops.object.camera_add(location=(5.0, -5.0, 4.0), rotation=(1.1, 0.0, 0.785))
    bpy.context.scene.camera = bpy.context.active_object

def setup_light():
    bpy.ops.object.light_add(type='SUN', location=(10.0, 10.0, 10.0))
    bpy.context.active_object.data.energy = 5.0
    bpy.ops.object.light_add(type='AREA', location=(-5.0, 5.0, 8.0))
    bpy.context.active_object.data.energy = 100.0

def import_model(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.location = (0.0, 0.0, 0.0)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    dim = max(obj.dimensions)
    if dim > 0:
        scale = 2.0 / dim
        obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(scale=True)

def render_single(obj_path, output_path):
    clear_scene()
    setup_render()
    setup_camera()
    setup_light()
    import_model(obj_path)
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def batch_render(csv_path, out_root):
    os.makedirs(out_root, exist_ok=True)
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        component_type = row['component_type']
        project_type = row['project_type']
        instance_path = row['instance_path']
        base_name = os.path.splitext(os.path.basename(instance_path))[0]
        iid = base_name.split('_')[-1]
        cat_dir = os.path.join(out_root, component_type, project_type)
        os.makedirs(cat_dir, exist_ok=True)
        png_path = os.path.join(cat_dir, f"{iid}.png")
        if not os.path.exists(png_path):
            try:
                render_single(instance_path, png_path)
                print("Rendered:", png_path)
            except Exception as e:
                print("Failed:", png_path, "Error:", str(e))

if __name__ == "__main__":
    CSV_FILE = r"D:\BIM_Component_Recognition\metadata\relabel_kept.csv"
    OUT_DIR = r"D:\BIM_Component_Recognition\datasets\former"
    batch_render(CSV_FILE, OUT_DIR)