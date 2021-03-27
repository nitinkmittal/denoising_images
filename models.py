import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Tuple

def conv2d_block(
    in_channels: Tuple[int, int], 
    out_channels: Tuple[int, int], 
    kernel_size: Tuple[int, int], 
    stride: Tuple[int, int], 
    padding: Tuple[int, int]):
  return nn.Sequential(
      nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,  
            stride=stride, 
            padding=padding),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(num_features=out_channels),
      nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(inplace=True),
  )


class ResNet32(nn.Module):
  def __init__(self):
    super().__init__()

    kernel_size=(3,3)
    stride=(1,1)
    padding=(1,1)

    # (_, 3, 32, 32) -> (_, 32, 32, 32)
    self.conv = nn.Conv2d(
        in_channels=3, 
        out_channels=32, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding)
    
    # (_ , 32, 32, 32) -> (_, 64, 32, 32)
    self.block1 = conv2d_block(
        in_channels=32, 
        out_channels=64, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding)
    
    # (_, 32, 32, 32) -> (_, 64, 32, 32)
    self.conv1 = nn.Conv2d(
        in_channels=32,
        out_channels=64, 
        kernel_size=(1,1), 
        stride=(1,1))
    
    # (_ , 64, 32, 32) -> (_, 128, 32, 32)
    self.block2 = conv2d_block(
        in_channels=64, 
        out_channels=128, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding) 
    
    # (_, 64, 32, 32) -> (_, 128, 32, 32)
    self.conv2 = nn.Conv2d(
        in_channels=64,
        out_channels=128, 
        kernel_size=(1,1), 
        stride=(1,1))
    
    # (_ , 128, 32, 32) -> (_, 64, 32, 32)
    self.block3 = conv2d_block(
        in_channels=128, 
        out_channels=64, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding) 
    
    # (_, 128, 32, 32) -> (_, 64, 32, 32)
    self.conv3 = nn.Conv2d(
        in_channels=128,
        out_channels=64, 
        kernel_size=(1,1), 
        stride=(1,1))
    
    # (_ , 64, 32, 32) -> (_, 32, 32, 32)
    self.block4 = conv2d_block(
        in_channels=64, 
        out_channels=32, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding) 
    
    # (_, 64, 32, 32) -> (_, 32, 32, 32)
    self.conv4 = nn.Conv2d(
        in_channels=64,
        out_channels=32, 
        kernel_size=(1,1), 
        stride=(1,1))
    
    # (_, 32, 32, 32) -> (_, 3, 32, 32)
    self.conv5 = nn.Conv2d(
        in_channels=32,
        out_channels=3, 
        kernel_size=(1,1), 
        stride=(1,1))
      
  def forward(self, x):
    
    # (_, 3, 32, 32) -> (_, 32, 32, 32)
    conv = F.relu(self.conv(x))

    # (_, 32, 32, 32) -> (_, 64, 32, 32)
    block = self.block1(conv)

    # (_, 32, 32, 32) -> (_, 64, 32, 32) + (_, 64, 32, 32) -> (_, 64, 32, 32) 
    add_block = F.relu(self.conv1(conv)) + block

    # (_, 64, 32, 32) -> (_, 128, 32, 32)
    block = self.block2(add_block)

    # (_, 64, 32, 32) -> (_, 128, 32, 32) + (_, 128, 32, 32) -> (_, 128, 32, 32)
    add_block = F.relu(self.conv2(add_block)) + block

    # (_, 128, 32, 32) -> (_, 64, 32, 32)
    block = self.block3(add_block)

    # (_, 128, 32, 32) -> (_, 64, 32, 32) + (_, 64, 32, 32) -> (_, 64, 32, 32)
    add_block = F.relu(self.conv3(add_block)) + block

    # (_, 64, 32, 32) -> (_, 32, 32, 32)
    block = self.block4(add_block)

    # (_, 64, 32, 32) -> (_, 32, 32, 32) + (_, 32, 32, 32) -> (_, 32, 32, 32)
    add_block = F.relu(self.conv4(add_block)) + block

    # (_, 32, 32, 32) -> (_, 3, 32, 32)
    return F.relu(self.conv5(add_block))


class CNN32(nn.Module):
  def __init__(self):
    super().__init__()

    kernel_size=(3,3)
    stride=(1,1)
    padding=(1,1)

    # (_, 3, 32, 32) -> (_, 32, 32, 32)
    self.conv = nn.Conv2d(
        in_channels=3, 
        out_channels=32, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding)
    
    # (_ , 32, 32, 32) -> (_, 64, 32, 32)
    self.block1 = conv2d_block(
        in_channels=32, 
        out_channels=64, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding)
    
    # (_ , 64, 32, 32) -> (_, 128, 32, 32)
    self.block2 = conv2d_block(
        in_channels=64, 
        out_channels=128, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding) 
    
    # (_ , 128, 32, 32) -> (_, 64, 32, 32)
    self.block3 = conv2d_block(
        in_channels=128, 
        out_channels=64, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding) 
    
    # (_ , 64, 32, 32) -> (_, 32, 32, 32)
    self.block4 = conv2d_block(
        in_channels=64, 
        out_channels=32, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding) 
    
    # (_, 32, 32, 32) -> (_, 3, 32, 32)
    self.conv5 = nn.Conv2d(
        in_channels=32,
        out_channels=3, 
        kernel_size=(1,1), 
        stride=(1,1))
      
  def forward(self, x):
    
    # (_, 3, 32, 32) -> (_, 32, 32, 32)
    conv = F.relu(self.conv(x))

    # (_, 32, 32, 32) -> (_, 64, 32, 32)
    block = self.block1(conv)

    # (_, 64, 32, 32) -> (_, 128, 32, 32)
    block = self.block2(block)

    # (_, 128, 32, 32) -> (_, 64, 32, 32)
    block = self.block3(block)

    # (_, 64, 32, 32) -> (_, 32, 32, 32)
    block = self.block4(block)

    # (_, 32, 32, 32) -> (_, 3, 32, 32)
    return F.relu(self.conv5(block))