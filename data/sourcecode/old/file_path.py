import glob, os
import io,sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = '../movie/get_file_name/*'

# パス内の全てのファイル・フォルダ名を取得
file_list = glob.glob(path)

# ファイル名だけを抽出
file_list = [os.path.basename(r) for r in file_list]

print(file_list[0])
