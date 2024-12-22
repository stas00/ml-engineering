# README - 中文翻译

# 构建书籍

重要：这仍然是进行中的工作——它基本可以运行，但样式表需要一些调整才能使PDF看起来更好。几周内应该会完成。

本文件假设您正在从存储库的根目录开始操作。

## 安装要求

1. 安装构建书籍过程中使用的Python包
````
pip install -r build/requirements.txt
```

2. 下载免费版本的[Prince XML](https://www.princexml.com/download/)。它用于构建本书的PDF版本。


## 构建HTML

````
make html
```

## 构建PDF

````
make pdf
``

它将首先构建HTML目标，然后使用该目标来构建PDF版本。


## 检查链接和锚点

要验证所有本地链接和锚点链接是否有效，请运行：
````
make check-links-local
``

要额外检查外部链接，请运行
````
make check-links-all
``

但请谨慎使用后者，以避免因频繁请求服务器而被封禁。


## 移动md文件/目录并调整相对链接

例如，将`slurm`移动到`orchestration/slurm`
````
src=slurm
dst=orchestration/slurm

mkdir -p orchestration
git mv $src $dst
perl -pi -e "s|$src|$dst|" chapters-md.txt
python build/mdbook/mv-links.py $src $dst
git checkout $dst
make check-links-local
``

## 调整图片大小

当包含的图片太大时，将其缩小一点：

````
mogrify -format png -resize 1024x1024\> *png
```