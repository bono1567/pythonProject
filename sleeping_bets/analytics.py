import json
from typing import Dict, List
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
np.seterr(all='ignore')


def calculate_vif(df):
    vif_data = pd.DataFrame()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

def remove_high_vif(df, threshold=5.0):
    while True:
        vif_data = calculate_vif(df)
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            df = df.drop(columns=[feature_to_remove])
        else:
            break
    return df


class Analytics:
    __all_columns = ['goals', 'yellowCards', 'redCards', 'groundDuelsWon',
       'groundDuelsWonPercentage', 'aerialDuelsWon',
       'aerialDuelsWonPercentage', 'successfulDribbles',
       'successfulDribblesPercentage', 'tackles', 'assists',
       'accuratePassesPercentage', 'totalDuelsWon', 'totalDuelsWonPercentage',
       'minutesPlayed', 'wasFouled', 'fouls', 'dispossessed', 'appearances',
       'saves', 'savedShotsFromInsideTheBox', 'savedShotsFromOutsideTheBox',
       'goalsConcededInsideTheBox', 'goalsConcededOutsideTheBox', 'accurateFinalThirdPasses',
       'bigChancesCreated', 'accuratePasses', 'keyPasses', 'accurateCrosses',
       'accurateCrossesPercentage', 'accurateLongBalls',
       'accurateLongBallsPercentage', 'interceptions', 'clearances',
       'dribbledPast', 'bigChancesMissed', 'totalShots', 'shotsOnTarget',
       'blockedShots', 'goalConversionPercentage', 'hitWoodwork', 'offsides',
       'expectedGoals', 'errorLeadToGoal', 'errorLeadToShot', 'passToAssist',
       'player', 'team', 'player id', 'team id']

    def __init__(self, data):
        self.__player_data = data[self.__all_columns].fillna(0)
        self.__scaler = StandardScaler()
    
    def __set_linear_y(self, linear_region_y, f_test_elimination_value):
        self.__linear_region_y = linear_region_y
        self.__f_test_elimination_value = f_test_elimination_value
        self.__drop_columns_for_regression = ['player', 'team', 'player id', 'team id', linear_region_y]
    
    def get_top_colinearity_to_linear_y(self) -> pd.Series:
        if not self.__linear_region_y:
            raise("The Linear regression Y not set")

        correaltion_matrix = self.__player_data.drop(['team', 'player'], axis=1).corr().dropna(how="any", axis=0).dropna(how="any", axis=1)
        return correaltion_matrix[self.__linear_region_y].sort_values(ascending=False)
    
    def __remove_multicollinearity(self) -> pd.DataFrame:
        return remove_high_vif(self.__player_data.drop(self.__drop_columns_for_regression, axis=1))
    
    def return_model(self):
        return self.__robust_model
    
    def p_value_test_summary(self) -> Dict:
        features_and_pvalues = []

        for feature, pval in zip(self.__regression_columns, self.__robust_model.pvalues):
            features_and_pvalues.append((feature, pval))

        print("Statistically Signifcant 99.9%")
        print([x for x in features_and_pvalues if x[1] < 0.001])

        print("Statistically Signifcant 99.5%")
        print([x for x in features_and_pvalues if x[1] < 0.05])

        print("Statistically Signifcant 99.9%")
        print([x for x in features_and_pvalues if x[1] < 0.1])

        return {
            1: features_and_pvalues,
            0.001: [x for x in features_and_pvalues if x[1] < 0.001],
            0.05: [x for x in features_and_pvalues if x[1] < 0.05],
            0.1: [x for x in features_and_pvalues if x[1] < 0.1]
        }
    
    def f_test_elimination_features(self, features_and_pvalues):
        # Get the highest 20 pvalue features and run a F-test
        lowest_all_features = [x[0] for x in features_and_pvalues[::-1]][:self.__f_test_elimination_value]
        argument = " = ".join(lowest_all_features) + " = 0"
        print("The null hypothesis: {}".format(argument))
        f_test_result = self.__robust_model.f_test(argument)


        if f_test_result.pvalue < 0.001:
            print("99% Null Hypothesis does not hold. The features do make a significant contribution")
        elif f_test_result.pvalue < 0.05:
            print("95% Null Hypothesis does not holds. The features do make a contribution")
        elif f_test_result.pvalue < 0.1:
            print("90% Null Hypothesis does not holds. The features do make a decent contribution")
        else:
            print("NULL Hypothesis hold. No significant contribution by the features")
        return f_test_result


    def run_regression_and_feature_selection(self, linear_region_y, f_test_elimination_value) -> List[str]:
        self.__set_linear_y(linear_region_y, f_test_elimination_value)
        print("Removing multicollinearity from dataset....")
        cleaned_data = self.__remove_multicollinearity()
        regression_columns = list(set(cleaned_data.columns) - set(self.__drop_columns_for_regression))
        self.__regression_columns = regression_columns

        print("Running OLS regression....")
        model = sm.OLS(self.__player_data[linear_region_y], self.__player_data[regression_columns]).fit()
        self.__robust_model = model.get_robustcov_results(cov_type='HC1')
        print(self.__robust_model.summary())

        # P-values of each feature
        print("Running p-value test...")
        features_and_pvalues = self.p_value_test_summary()[1]

        # F-test on highest p-values
        print("Running f-test...")
        self.f_test_elimination_features(features_and_pvalues=features_and_pvalues)

        features_and_pvalues.sort(key=lambda x: x[1])
        final_features = [x[0] for x in features_and_pvalues[::-1]][:-f_test_elimination_value]
        print("Final features for predicting: {}".format(linear_region_y))
        print(final_features)
        final_features.extend(self.__drop_columns_for_regression)
        print("The prediction variable: {}".format(final_features[-1]))
        return final_features
    
    def save_most_similar_players_list(self, file_path, n, final_prediction_data):
        final_prediction_data = final_prediction_data.reset_index(drop=True)
        scaled_df = self.__scaler.fit_transform(final_prediction_data.drop(self.__drop_columns_for_regression, axis=1))

        # Imporve similarity algorithm
        similarities = cosine_similarity(scaled_df)
        np.fill_diagonal(similarities, -1)
        top_n_similar = np.argsort(similarities, axis=1)[:, -n:]

        # Get scores for top `n` similar players
        similarity_scores = np.sort(similarities, axis=1)[:, -n:]  # For similarity

         # Write output to the file
        list_of_player_similarities = {}
        for i, (indices, scores) in enumerate(zip(top_n_similar, similarity_scores)):
            player_name = final_prediction_data.loc[i, 'player']
            similar_players = final_prediction_data.loc[indices, 'player'].values
            list_of_player_similarities[player_name] = []
            for sim_player, score in zip(similar_players, scores):
                actual_value = int(final_prediction_data[final_prediction_data['player'] == sim_player][self.__linear_region_y].iloc[0])
                team = final_prediction_data[final_prediction_data['player'] == sim_player]['team'].iloc[0]
                list_of_player_similarities[player_name].append({'player': sim_player,
                                                                 'team' : team,
                                                                 'score': score,
                                                                 self.__linear_region_y: actual_value})
        
        with open(file_path, 'w', encoding='utf-8') as fs:
            json.dump(list_of_player_similarities, fs, ensure_ascii=False, indent=4)
    
    def predict(self, data_point) -> pd.Series:
        data_point = data_point[self.__regression_columns].fillna(0)
        return self.__robust_model.predict(data_point)
        

class ClubAnalytics:
    def __init__(self, all_data):
        self.__all_data = all_data.reset_index(drop=True)
    
    def get_club_of_players(self, player_list: List[str]) -> List[str]:
        list_of_clubs = []
        for player in player_list:
            list_of_clubs.append(self.__all_data[self.__all_data["player"] == player]['team'].values[0])
        return list_of_clubs
    
    def get_players_of_club(self, club_name:str) -> pd.DataFrame:
        return self.__all_data[self.__all_data['team'] == club_name]

    
















    


        