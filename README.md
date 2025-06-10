# [NTIRE 2025 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## How to test the model?

1. `git clone https://github.com/zhengchen1999/NTIRE2025_ImageSR_x4.git`
2. Download the model weights from:

    - [Baidu Pan](https://pan.baidu.com/s/1iqaonrwEQVTbqp-1IcrhAg?pwd=SRSR) (validation code: **SRSR**)
    - [Google Drive](https://drive.google.com/drive/folders/18ePdU3ZZO3Tk9meqSmP-Yrkv-OU-RLbE?usp=drive_link)

    Put the downloaded weights in the `./model_zoo` folder.
3. Select the model you would like to test:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
    - We provide a baseline (team00): DAT (default). Switch models (default is DAT) through commenting the code in [test.py](./test.py#L19).
4. We also provide the output of each team from:

    - [Baidu Pan](https://pan.baidu.com/s/1Ah6il9Sfe3hkRP8_Nv5KXw?pwd=SRSR) (validation code: **SRSR**)
    - [Google Drive](https://drive.google.com/drive/folders/1R32G2xEWh-igZTkEvpcMg7jX5BGqSQeg?usp=drive_link)

    You can directly download the output of each team and evaluate the model using the provided script.
5. Some methods cannot be integrated into our codebase. We provide their instructions in the corresponding folder. If you still fail to test the model, please contact the team leaders. Their contact information is as follows:

| Index |       Team      |            Leader            |              Email              |
|:-----:|:---------------:|:----------------------------:|:-------------------------------:|
|   1   | SamsungAICamera |         Xiangyu Kong         |     xiangyu.kong@samsung.com    |
|   2   |      SNUCV      |         Donghun Ryou         |         dhryou@snu.ac.kr        |
|   3   |       BBox      |            Lu Zhao           |       zlcossiel@gmail.com       |
|   4   |     XiaomiMM    |          Hongyuan Yu         |      yuhyuan1995@gmail.com      |
|   5   |     MicroSR     |          Yanhui Guo          |       guoy143@mcmaster.ca       |
|   6   |     NJU_MCG     |            Xin Liu           |   xinliu2023@smail.nju.edu.cn   |
|   7   |       X-L       |           Zeyu Xiao          |    zeyuxiao@mail.ustc.edu.cn    |
|   8   |    Endeavour    |        Yinxiang Zhang        |  zhangyinxiang@mail.nwpu.edu.cn |
|   9   |   KLETech-CEVI  | Vijayalaxmi Ashok Aralikatti |    01fe21bcs181@kletech.ac.in   |
|   10  |     CidautAi    |        Marcos V. Conde       |  marcos.conde@uni-wuerzburg.de  |
|   11  |      JNU620     |          Weijun Yuan         |    yweijun@stu2022.jnu.edu.cn   |
|   12  |     CV_SVNIT    |          Aagam Jain          |    aagamjainaj1805@gmail.com    |
|   13  |      ACVLAB     |         Chia-Ming Lee        |      zuw408421476@gmail.com     |
|   14  |     HyperPix    |      Risheek V Hiremath      |   hiremathrisheek745@gmail.com  |
|   15  |      BVIVSR     |         Yuxuan Jiang         |      dd22654@bristol.ac.uk      |
|   16  |      AdaDAT     |         Jingwei Liao         |          jliao2@gmu.edu         |
|   17  |      Junyi      |          Junyi Zhao          |      z15236936309@gmail.com     |
|   18  |     ML_SVNIT    |          Ankit Kumar         |    ankitkumar735226@gmail.com   |
|   19  |     SAK_DCU     |      Sunder Ali Khowaja      |     sunderali.khowaja@dcu.ie    |
|   20  |      VAI-GM     |      Snehal Singh Tomar      |     stomar@cs.stonybrook.edu    |
|   21  |   Quantum Res   |       Sachin Chaudhary       | sachin.chaudhary@ddn.upes.ac.in |
|   22  |       PSU       |        Bilel Benjdira        |       bbenjdira@psu.edu.sa      |
|   23  |    IVPLAB-sbu   |        Zahra Moammeri        |     zahramoammeri1@gmail.com    |
|   24  |      MCMIR      |          Liangyan Li         |        lil61@mcmaster.ca        |
|   25  |     Aimanga     |         Zonghao Chen         |  chenzonghao@k-fashionshop.com  |
|   26  |       IPCV      |      Jameer Babu Pinjari     |       jameer.jb@gmail.com       |

## How to eval images using IQA metrics?

### Environments

```sh
conda create -n NTIRE-SR python=3.8
conda activate NTIRE-SR
pip install -r requirements.txt
```


### Folder Structure
```
test_dir
├── HR
│   ├── 0901.png
│   ├── 0902.png
│   ├── ...
├── LQ
│   ├── 0901x4.png
│   ├── 0902x4.png
│   ├── ...
    
output_dir
├── 0901x4.png
├── 0902x4.png
├──...

```

### Command to calculate metrics

```sh
python eval.py \
--output_folder "/path/to/your/output_dir" \
--target_folder "/path/to/test_dir/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```

The `eval.py` file accepts the following 4 parameters:
- `output_folder`: Path where the restored images are saved.
- `target_folder`: Path to the HR images in the `test` dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

### Weighted score for Perception Quality Track

We use the following equation to calculate the final weight score: 

$$
\text{Score} = \left(1 - \text{LPIPS}\right) + \left(1 - \text{DISTS}\right) + \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right). 
$$

The score is calculated on the averaged IQA scores. 

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
