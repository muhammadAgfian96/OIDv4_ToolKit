import os


class GeneratorConfigsOIDv6:
    pass
# for downloader
args_downloader = {
    'command' : 'downloader',
    'classes' : ['Apple', 'Human_head'],
    'multiclasses': 0, # 1 if you wan all class in one folder, 0 for seperate in each folder base in class 
    'sub' : 'h',
    'type_csv': 'train',

    'image_IsOccluded': '',
    'image_IsTruncated': '', 
    'image_IsGroupOf' : 0, 
    'image_IsDepiction': 0, 
    'image_IsInside': '',

    'n_threads': 6,
    'limit': 100,
}



def generate_command(all_args):
    my_args = ''
    for key in all_args.keys():
        value = all_args[key]
        if value == '':
            continue

        if key == 'command':
            my_args += f' {value}'
            continue

        if key == 'classes':
            my_args+= f' --{key}'
            for classes in value:
                my_args += f' {classes}'
            continue


        my_args += f' --{key} {value}'

    return my_args

my_args = generate_command(args_downloader)
print('#1. copy this command and execute')
print(f'!python3 main.py'+my_args)