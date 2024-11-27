# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:37:39 2024

@author: jspark
"""

from data import load_smiles, Dataloader
from utils import output, merge, applicability_domain_test, applicability_domain_average
from model_utils import load_config, load_model
from predict import predict


def main(script_path, input_path, output_path, KE):

    if KE not in [1, 2, 3, 4, 'All']:
        raise Exception('KE is wrong (Only 1, 2, 3, 4, or All)')
    
    KE_list = [KE]
    if KE == 'All':
        KE_list = [1,2,3,4]
    
    model_path = f'{script_path}/model'
    
    # load smiles
    smiles_list = load_smiles(input_path)
    
    df_list = []
    for KE in KE_list:
        print(f'KE{KE} Prediction Start!!!')
        # load configuration
        config = load_config(model_path, KE)
        
        # load dataloader
        dataloader = Dataloader(smiles_list, config)
        
        # load model
        model = load_model(model_path, config, KE)
        
        
        # make prediction
        probs, labels, test_vectors_mf, test_vectors_lf = predict(model, dataloader)

        # when evaluate
        # from utils import evaluate
        # import pandas as pd
        # df = pd.read_excel(input_path)
        # y_list = df['cat_label']
        # evaluate(y_list, labels, probs)
        
        # Applicability Domain Test
        # mf : GCN feature + molecular feature
        # lf : last feature
        ad_mf, ad_lf = applicability_domain_test(KE, script_path, test_vectors_mf, test_vectors_lf) # 학습 데이터와 최소거리 계산
        # ad_mf, ad_lf = applicability_domain_average(KE, script_path, test_vectors_mf, test_vectors_lf) # 학습 데이터 전체와 평균거리 계산
        
        # make output file
        # AD = 'all' : mf, lf 둘 다 사용
        # AD = 1 : mf 사용
        # AD = 2 : lf 사용
        df_output = output(output_path, smiles_list, labels, probs, ad_mf, ad_lf, KE, save_proba = True, AD = 'all')

        df_list.append(df_output)
        
        print(f'KE{KE} Prediction Finished\n')
    
    merged_df = merge(output_path, df_list)
    
    return merged_df


# script version  
if __name__ == '__main__':
    
    KE = 'All'  # 1, 2, 3, 4, All
    # KE = 1
    script_path = r'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script'
    input_path = r'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script\examples\KE1_test_examples.xlsx'
    # input_path = r'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script\examples\test.xlsx'
    output_path = rf'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script\output\test_{KE}_output.xlsx'
    # output_path = r'C:\Users\user\Desktop\1\Modeling\18. AOP 예측모델\피부과민성\통합 script\output\test.xlsx'
    
    df = main(script_path, input_path, output_path, KE)
    
    
# argument version

# import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='AOP 예측 모델 스크립트')
    
#     # 명령줄 인수 정의
#     parser.add_argument('--model_path', type=str, required=True, help='모델 경로')
#     parser.add_argument('--input_path', type=str, required=True, help='입력 파일 경로')
#     parser.add_argument('--output_path', type=str, required=True, help='출력 파일 경로')
#     parser.add_argument('--KE', type=int, default=2, help='KE 값 (기본값: 2)')
    
#     # 인수 파싱
#     args = parser.parse_args()
    
#     # main 함수 호출
#     df = main(args.model_path, args.input_path, args.output_path, args.KE)