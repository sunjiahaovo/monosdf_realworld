#render video from the images in render_exps

import os
import argparse
import imageio.v2 as imageio

def render_video(expname, out_path):
    images = []
    for file_name in sorted(os.listdir(os.path.join('../render_exps', expname, 'plots', 'test_all'))):
        if file_name.startswith('rendering'):
            if len(file_name) == 19:
                print('reading: ',file_name)
                images.append(imageio.imread(os.path.join(expname, 'plots', 'test_all', file_name)))
                # test = imageio.imread(os.path.join(expname, 'plots', 'test_all', file_name))
    for file_name in sorted(os.listdir(os.path.join('../render_exps', expname, 'plots', 'test_all'))):
        if file_name.startswith('rendering'):        
            if len(file_name) == 20:
                print('reading: ',file_name)
                images.append(imageio.imread(os.path.join(expname, 'plots', 'test_all', file_name)))
    for file_name in sorted(os.listdir(os.path.join('../render_exps', expname, 'plots', 'test_all'))):
        if file_name.startswith('rendering'):                
            if len(file_name) == 21:
                print('reading: ',file_name)
                images.append(imageio.imread(os.path.join(expname, 'plots', 'test_all', file_name)))
    
    #set fps
    fps = 20
    with imageio.get_writer(out_path, fps=fps) as video:
        for image in images:
            video.append_data(image)
            
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, default='')
    
    opt = parser.parse_args()
    print('opt.expname: ', opt.expname)
    
    expname = os.path.join('../render_exps', opt.expname)
    outpath = os.path.join('../render_exps', opt.expname, 'rendering.mp4')
    
    render_video(expname, outpath)
    
    
    
    
    
