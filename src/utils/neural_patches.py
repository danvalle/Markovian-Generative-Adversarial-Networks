import torch


class NeuralPatches():

    def __init__(self, block_size, block_stride, num_output_plane,
                 flag_all, device):
        super(NeuralPatches, self).__init__()
        self.block_size = block_size
        self.block_stride = block_stride
        self.num_output_plane = num_output_plane
        self.flag_all = flag_all
        self.device = device

    def compute_patches(self, target_copies, net):
        x_per_image, y_per_image, target_image_len = self.target_per_image(
            target_copies, net)
        print('Number of target patches: {}'.format(
            target_image_len))
        
        target_patches = torch.Tensor(
            target_image_len,
            self.num_output_plane,
            self.block_size,
            self.block_size)
        target_patches = target_patches.to(device=self.device)
        
        patch_num = 0
        for copy_idx, target_copie in enumerate(target_copies):
            feature_map = net(target_copie.unsqueeze(0)).clone()
            
            x_per_image_len = x_per_image[copy_idx].size()[0]
            for patch_per_image in range(x_per_image_len):
                y_start = int(
                    y_per_image[copy_idx][patch_per_image].item())
                y_end = y_start + self.block_size
                x_start = int(
                    x_per_image[copy_idx][patch_per_image].item())
                x_end = x_start + self.block_size

                target_patches[patch_num] = feature_map[
                    0, :, y_start:y_end, x_start:x_end]
                patch_num = patch_num + 1

        return target_patches

    def target_per_image(self, target_copies, net):
        target_x_per_image = []
        target_y_per_image = []
        target_image_len = 0
        for target_copie in target_copies:
            feature_map = net(target_copie.unsqueeze(0)).clone()

            target_x, target_y = self.compute_grid(
                feature_map.size()[3], feature_map.size()[2])

            target_size = len(target_x) * len(target_y)
            x_per_image = torch.Tensor(target_size)
            y_per_image = torch.Tensor(target_size)
            count = 0
            for row in range(len(target_y)):
                for col in range(len(target_x)):
                    x_per_image[count] = target_x[col]
                    y_per_image[count] = target_y[row]
                    count += 1
            
            target_x_per_image.append(x_per_image)
            target_y_per_image.append(y_per_image)
            
            target_image_len += target_size

        return target_x_per_image, target_y_per_image, target_image_len

    def coord_block(self, dim):
        coord_block = torch.arange(0, dim-self.block_size, self.block_stride)
        if self.flag_all:
            if coord_block[-1] < dim - self.block_size:
                tail = torch.LongTensor(1)
                tail[0] = dim - self.block_size
                coord_block = torch.cat((coord_block, tail))
        return coord_block

    def compute_grid(self, width, height):
        coord_block_y = self.coord_block(height)
        coord_block_x = self.coord_block(width)
        return coord_block_x, coord_block_y