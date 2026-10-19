# 実行ガイド：Windows / Linux

## 1. 単独実行に必要なもの

`research_mvp` は、過去の研究ディレクトリから独立して実行できます。必要なのは、本ディレクトリ内のソースコード、データセット、Pythonの依存パッケージ、および初期化用の重みです。親ディレクトリにある過去のPythonスクリプト、ログ、論文、旧実験の重みには依存しません。

対応する処理フローは以下のみです。

```text
yolov8s.ptの事前学習済み重みで初期化
→ train/val全体を用いた学習・検証 → 今回の学習で得られたbest.pt
→ testを時系列順に推論 → Clean → Interpolate → 3段階を共通の評価器で評価
```

今回のスモークテストでは、学習画像8枚、検証画像4枚、テスト画像24枚のみを使用し、入力サイズ320で1 epochだけ学習しています。そのため、通常の学習性能を示すものではありません。train/val全体、入力サイズ1280、50または100 epochsを使用する設定は、今後実施する新たな実験の設定です。データ数や学習回数を増やしても必ず性能が改善するとは限らないため、検証セットおよび独立したテストセットの結果に基づいて判断してください。

**本ガイドに記載した全量学習コマンドは、今回の作業では実行していません。** 実行確認済みなのは、WindowsのCPU環境における小規模な1 epochの処理フローのみです。Linux向けの手順は、現在の実行インターフェースとパス処理に基づいて記載していますが、実機での動作は未確認です。

## 2. データセットのダウンロードと配置

データセット：

[Badminton360-Shuttlecock — Hugging Face](https://huggingface.co/datasets/aissur320/Badminton360-Shuttlecock)

上記からデータセットをダウンロードして展開し、データディレクトリ名を **`project`** にしてください。**`research_mvp` と同じ親ディレクトリ内**に配置します。

```text
your_workspace/
├── project/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── research_mvp/
    ├── mvp.py
    ├── fetch_weights.py
    ├── configs/
    ├── src/
    ├── tests/
    └── ...
```

`images/` と `labels/` が `project/` の直下にあり、同名ディレクトリが余分に入れ子になっていないことを確認してください。学習・検証・テストの各画像には、拡張子を除くファイル名が同じTXTラベルファイルが必要です。背景画像についても、空のラベルファイルを用意してください。

外部の `project/` は元データです。後述の `prepare-data` は、全データを `research_mvp/data/project/` にコピーし、確認済みのクラス0と3を、**コピー内でのみ**羽毛球クラス0に統合します。元データは変更しません。コピーに必要なディスク容量を確保してください。

## 3. Windows：環境構築

PowerShellを使用し、正常に動作するPython 3.12を事前にインストールしてください。`project/` と `research_mvp/` があるディレクトリでターミナルを開き、MVPディレクトリに移動します。

```powershell
cd research_mvp
python -m venv .venv
```

`.venv` は各端末で新しく作成してください。他の端末からコピーした仮想環境は再利用しないでください。以下のコマンドでは仮想環境内のPythonを直接指定するため、仮想環境の有効化は不要です。

### CPUを使用する場合

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

CPUで入力サイズ1280の全量学習を行う場合、長時間かかる可能性があります。利用可能なNVIDIA GPUがある場合は、以下のGPU向け手順を選択できます。

### NVIDIA GPUを使用する場合

新しい仮想環境で、ドライバーとプラットフォームに適合するPyTorchのCUDA版インストールコマンドを選択してください。

[PyTorch公式：過去バージョンのインストール手順](https://pytorch.org/get-started/previous-versions/)

互換性のある組み合わせである `torch==2.7.1` と `torchvision==0.22.1` をインストールした後、以下を実行します。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

最後の出力が `True` であることを確認してから、第5節で `device: "0"` を設定してください。CPU向けとGPU向けのインストール方法は、どちらか一方を選択してください。GPUを使用する環境に、先に `requirements-cpu.txt` をインストールしないでください。

### 依存関係の確認と初期化用重みのダウンロード

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe fetch_weights.py
```

ダウンロード後、`weights/yolov8s.pt` が存在することを確認してください。信頼できる入手元のファイルがすでに存在する場合は、ダウンロードを省略してください。ダウンロード用スクリプトは既存ファイルを上書きしません。

## 4. Linux：環境構築

Bashを使用し、Python 3.12と、そのバージョンに対応する `venv` を準備してください。`project/` と `research_mvp/` があるディレクトリでターミナルを開きます。以下では、`python3` が必要なバージョンを指しているものとします。

```bash
cd research_mvp
python3 --version
python3 -m venv .venv
```

### CPUを使用する場合

```bash
./.venv/bin/python -m pip install -r requirements-cpu.txt
```

### NVIDIA GPUを使用する場合

[PyTorch公式の過去バージョン向けインストール手順](https://pytorch.org/get-started/previous-versions/)に従い、新しい仮想環境に互換性のあるCUDA版の `torch==2.7.1` と `torchvision==0.22.1` をインストールしてください。その後、以下を実行します。

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

GPUを使用する場合、CUDAの利用可否の確認結果が `True` である必要があります。CPU向けとGPU向けのインストール方法は、どちらか一方を選択してください。

### 依存関係の確認と初期化用重みのダウンロード

```bash
./.venv/bin/python -m pip check
./.venv/bin/python fetch_weights.py
```

信頼できる `weights/yolov8s.pt` がすでに存在する場合は、ダウンロードを省略してください。他のモデルの重みを `yolov8s.pt` に名前変更して使用しないでください。

## 5. 全量学習用の設定ファイルを作成する

必要な項目を含む既存の設定例をコピーし、ローカル設定ファイルを作成します。

Windows：

```powershell
Copy-Item configs/experiment.example.json configs/local.full.json
```

Linux：

```bash
cp configs/experiment.example.json configs/local.full.json
```

テキストエディターで `configs/local.full.json` を開き、**BOMなしのUTF-8**で保存してください。他の項目を維持したまま、使用する端末に合わせて以下を設定します。

| 項目 | 設定内容 |
|---|---|
| `run_name` | `yolov8s_full_50e_cpu` または `yolov8s_full_50e_gpu` など、新しい一意の名前を指定します。 |
| `paths.data_root` | インポートしたデータのコピー先として `data/project` を維持します。 |
| `paths.pretrained_weights` | `weights/yolov8s.pt` を維持します。 |
| `paths.output_root` | `outputs` を維持するか、MVPディレクトリ内でデータと分離した新しい出力ディレクトリを指定します。 |
| `train.epochs` | `50` に設定します。100 epochsの場合は `100` にします。設定例の初期値は100です。 |
| `train.imgsz` | `1280` を維持します。 |
| `train.batch` | CPUではまず `2` を使用できます。GPUではVRAM容量に合わせて設定してください。設定例の初期値は16です。 |
| `train.device` | CPUの場合は `"cpu"`、最初の利用可能なNVIDIA GPUの場合は `"0"` を指定します。 |
| `predict.imgsz` | `1280` を維持します。 |
| `predict.device` | 選択した端末に合わせて `"cpu"` または `"0"` を指定します。 |

`epochs` や `batch` などは数値、`device` は文字列です。設定内のパス項目は `research_mvp/` を基準に解決されます。一方、`--config` に渡す設定ファイルの相対パスは、現在の作業ディレクトリを基準に解決されます。そのため、本ガイドでは常にMVPディレクトリ内でコマンドを実行します。

プログラムは実行時に現在のプロジェクト位置から絶対パスを解決するため、開発者の端末固有の固定パスを含みません。`config.resolved.json` と学習時の `args.yaml` に保存されるファイルパスは、`research_mvp/` を基準とする相対パスに統一され、区切り文字には `/` を使用します。`dataset.yaml` 内のtrain/valパスは、そのYAMLファイルの配置ディレクトリを基準とする相対パスであり、端末固有の `path` 項目は設定しません。プロジェクト全体を移動する際は、`data/`、`weights/`、`outputs/` の相対的なディレクトリ構造を維持してください。

新しく利用する場合は、設定例から自分の実行設定を作成してください。`outputs/` はGitの追跡対象外となるローカルの実行生成物です。`.venv/` と `.cache/` も公開や端末間コピーの対象にせず、移行先で前述の手順に従って依存パッケージを再インストールしてください。

`source_class_map` は `{"0": 0, "3": 0}` のままにしてください。ダウンロードしたデータに他のクラスIDが含まれる場合は、その意味を確認し、推測でマッピングしないでください。

**学習開始後は、推論デバイス、しきい値、`run_name` を含め、設定を変更しないでください。** 後続の処理では、学習時に保存した設定との一致を確認します。

## 6. Windows：全量データで実行する

以下では、第5節で作成した同一の設定ファイルを使用します。各処理の成功を確認してから、次の処理を実行してください。

```powershell
# 外部の元データを変更せずに検査し、MVP内に検査レポートを保存
.\.venv\Scripts\python.exe -X utf8 mvp.py audit --data-root ../project

# 全量データをコピーし、ラベルをマッピングする。この処理では学習を開始しない
.\.venv\Scripts\python.exe -X utf8 mvp.py prepare-data --source ../project --config configs/local.full.json

# train/val全体を使用して学習・検証。50/100 epochsには明示的なフラグが必要
.\.venv\Scripts\python.exe -X utf8 mvp.py train --config configs/local.full.json --allow-full-train

# 今回の学習で得られたbest.ptを読み込み、test全体を時系列順に推論
.\.venv\Scripts\python.exe -X utf8 mvp.py predict --config configs/local.full.json

# CleanとInterpolateの結果をそれぞれ保存
.\.venv\Scripts\python.exe -X utf8 mvp.py postprocess --config configs/local.full.json

# raw、Clean、Interpolateを共通の評価器で評価
.\.venv\Scripts\python.exe -X utf8 mvp.py evaluate --config configs/local.full.json
```

## 7. Linux：全量データで実行する

```bash
./.venv/bin/python -X utf8 mvp.py audit --data-root ../project
./.venv/bin/python -X utf8 mvp.py prepare-data --source ../project --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py train --config configs/local.full.json --allow-full-train
./.venv/bin/python -X utf8 mvp.py predict --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py postprocess --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py evaluate --config configs/local.full.json
```

同じコピー先データディレクトリに対する `prepare-data` の実行は、一度だけで十分です。`data/project/` が、すでに正常にインポートとラベルマッピングを完了した全量データのコピーである場合、以後の実験でも再利用し、再インポートしないでください。一部のみインポートされた状態の場合は、新しい `paths.data_root` を指定してから再度インポートし、上書きやデータの混在を避けてください。

全量学習に `run-smoke` を使用しないでください。このコマンドは学習を1 epochに固定し、サンプル数も制限します。全量学習には上記の段階別コマンドを使用してください。

## 8. 結果の確認と再実験

`run_name: "yolov8s_full_50e_gpu"` の場合、出力例は以下のとおりです。

```text
outputs/yolov8s_full_50e_gpu/
├── config.resolved.json
├── dataset_audit.json
├── manifest.json
├── training.json
├── train/
│   ├── args.yaml
│   ├── results.csv
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── raw.json
├── clean.json
├── interpolated.json
└── metrics.json
```

- `best.pt` は、学習中の検証セットでの性能に基づいて選択されます。推論では、その実行に記録された `best.pt` のみを受け付け、SHA-256ハッシュを確認します。過去のスモークテストの重みは使用しません。
- `metrics.json` には、3段階それぞれのTP/FP/FN、P/R@0.5、AP50、mAP50–95が含まれます。学習フレームワーク内部の検証指標と、この共通評価器の指標は定義が異なるため、同一の指標として扱わないでください。
- `manifest.json` には、シーケンスID、数値のフレーム番号、および順序が明示的に保存されます。欠落フレーム、フレーム番号の重複、順序の不整合、画像サイズの変化がある場合はエラーになります。後処理はシーケンスをまたいで行いません。
- CleanとInterpolateは、予測と使用が認められたフレーム情報のみを利用します。testの正解ラベルは最終評価のみに使用します。train/valのラベルは教師あり学習と検証に使用します。
- 本実装は、実際の画像幅と新しいAP101評価プロトコルを使用する新たな実験です。結果をそのまま論文の再現結果と呼ぶことはできません。

新しく学習するたびに、新しい `run_name` を使用してください。既存の出力は上書きできません。また、設定を変更した後に、以前の実行をそのまま使って推論や評価を行うこともできません。現在の実行インターフェースでは `resume=False` に固定されており、中断した学習を再開するコマンドは提供していません。学習中断後に再実験する場合は、実行名を変更してください。

VRAMまたはシステムメモリが不足した場合は、失敗した実行を終了し、新しい設定と新しい実行名で `train.batch` を小さくしてください。入力サイズを1280に維持することで、正式な実験を説明なくスモークテスト用の320に変更することを避けられます。入力サイズを変更する必要がある場合は、新しい実験条件として記録してください。

必要に応じて、既存の単体テストを実行できます。

Windows：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux：

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

`data/`、`weights/`、`outputs/`、`.venv/`、キャッシュ、および `configs/local*.json` は、MVPの `.gitignore` で除外されています。外部の同階層にある `project/` も公開リポジトリに追加しないでください。公開対象はMVPのソースコードのみです。

本ガイドは操作手順を示すものであり、ダウンロード、インストール、データのインポート、テスト、学習を自動的に実行するものではありません。
