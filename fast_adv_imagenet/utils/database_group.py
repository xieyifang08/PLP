import json
import shutil
import os

f = open('./database_group.json', 'r')
content = f.read()
data = json.loads(content)
print(data.keys())
# shutil.copyfile()
dicts = {
    "defence_fail_upper": "attention_move",
    "defence_fail_down": "attention_weak"
}

src_path = "/mnt/u2/code/PLP/fast_adv_imagenet/attacks/advs"
dst_path = "/mnt/u2/code/PLP/fast_adv_imagenet/attacks/advs/twoGroupDatabase"
for name in data.keys():
    dst = os.path.join(dst_path, name, dicts["defence_fail_upper"])
    if not os.path.exists(dst):
        os.makedirs(dst)
    for img in data[name]["defence_fail_upper"]:
        shutil.copy(os.path.join(src_path, name, img), os.path.join(dst, img))

    dst = os.path.join(dst_path, name, dicts["defence_fail_down"])
    if not os.path.exists(dst):
        os.makedirs(dst)
    for img in data[name]["defence_fail_down"]:
        shutil.copy(os.path.join(src_path, name, img), os.path.join(dst, img))
f.close()