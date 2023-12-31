# 開発ガイド

ここでは、開発に関する簡単な流れを説明します。

他のプロジェクトでも同様の流れを取れるようにします。

<br>

## 使用する環境

使用する環境はつぎの通りです。

### ハードウェア

- VSCodeが動くくらいの性能のPC

### ソフトウェア

- Ubuntu22.04 LTS (ROS 2 Humbleのrosdepが解決できる環境)
- ROS 2 Humble

<br>

## 開発の流れ

開発の流れはつぎの通りです。

1. タスクの割当てがある。（issue・Projectsのチケット）
    - このとき、タスクの割当ては、`~~_plugins` フォルダに対して行う。
    - テストコードも作成しておく
2. ブランチを切る。
3. `~~_plugins` フォルダに対して、ソースコードを作成する。
4. `~~_plugins` フォルダに対して、ドキュメントを作成する。
4. `~~_plugins` フォルダに対して、テストコードを作成する。（ない場合）
6. テストが通るようになったら、PRを出してレビューを受ける。
7. レビューが通ったら、マージする。

途中でパラメータの追加などの必要があれば、適宜追加してもらって構いません。

<br>


## ブランチの命名規則

ブランチの命名規則はつぎの通りです。

- `feat/<機能名>` : 機能追加
- `fix/<機能名>` : 機能修正
- `refactor/<機能名>` : 機能のリファクタリング
- `docs/<機能名>` : ドキュメントの追加・修正
- `test/<機能名>` : テストの追加・修正

<br>

## フォルダ構成

ここでは、Detector2dのフォルダ構成を例に説明します。

フォルダ構成はつぎの通りです。

![](./dev_guide.png)

ファイルとして作成するのは、 `detector2d_plugins` 内のヘッダーファイルとソースコードファイルのみ（橙色で囲っている部分）ですが、その登録先（青で囲っている部分）については新規作成した内容に応じて適宜修正していきます。

ヘッダーファイルについては、事前に作成されている`_base.hpp`を継承します。

テスト内容は、`_plugin.cpp`(ここでは`publish_center.cpp`）をテストします。

<br>

publish_centerの場合、新しく追加するファイルはつぎの通りです。

- `detector2d/detector2d_plugins/src/publish_center.cpp` : ソースコード
- `detector2d/detector2d_plugins/include/detector2d_plugins/publish_center.hpp` : ヘッダファイル
  - `detector2d_base.hpp` を継承します。

プラグインを追加するため、次のファイルを追加で記述します。（記述しないとコンパイルエラーになります。）

- `detector2d/detector2d_plugins/detector2d_plugins.xml`
- `detector2d/detector2d_plugins/CMakeLists.txt`

必要に応じて次のファイルを追加で記述します。

- `detector2d/detector2d_plugins/package.xml`
- `detector2d/build_depends.repos`


<br>

## テストについて

TODO

<br>

## ドキュメントについて

ドキュメントは、チーム外にも共有されます。

初めてでもわかりやすい説明を心がけましょう。

<br>



