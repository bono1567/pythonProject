import pandas as pd
import numpy as np
import ScraperFC as sfc
from analytics import Analytics, ClubAnalytics

ss = sfc.Sofascore()
np.seterr(all='ignore')

all_leagues = ['EPL', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1','Champions League', 'Europa League', 'Europa Conference League']
all_columns = ['goals', 'yellowCards', 'redCards', 'groundDuelsWon',
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

f_test_elimination_value = 10
season = "24/25"

player_data = []
for league in all_leagues[:5]:
    player_data.append(ss.scrape_player_league_stats(season, league))
    print("Fetched data for {} and season {}".format(league, season))
original_player_data = pd.concat(player_data)[all_columns].fillna(0).reset_index(drop=True)
player_data = pd.concat(player_data[2:5])[all_columns].fillna(0)

final_features = ['goals', 'assists', 'tackles', 'yellowCards', 'redCards', 'totalShots', 'shotsOnTarget', 'wasFouled', 'fouls', 'team', 'player']
analysis_goals = Analytics(player_data)
# final_features = analysis_goals.run_regression_and_feature_selection('goals', 10)
final_prediction_data = original_player_data[final_features]
club_analysis_goals = ClubAnalytics(original_player_data[final_features])
# analysis_goals.save_most_similar_players_list('similar_player_goals.json', 5, final_prediction_data)

analysis_assists = Analytics(player_data)
# final_features = analysis_assists.run_regression_and_feature_selection('assists', 10)
final_prediction_data = original_player_data[final_features]
club_analysis_assists = ClubAnalytics(original_player_data[final_features])
# analysis_assists.save_most_similar_players_list('similar_player_assists.json', 5, final_prediction_data)

analysis_fouls = Analytics(player_data)
# final_features = analysis_fouls.run_regression_and_feature_selection('fouls', 10)
final_prediction_data = original_player_data[final_features]
club_analysis_fouls = ClubAnalytics(original_player_data[final_features])
# analysis_fouls.save_most_similar_players_list('similar_player_fouls.json', 5, final_prediction_data)

# print(analysis_assists.predict(original_player_data[original_player_data['player'] == 'Cole Palmer']).to_list()[0])
# print(original_player_data[original_player_data['player'] == 'Cole Palmer']['assists'].to_list()[0])

import solara
import json

# Reactive state variables
leagues = ['EPL', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
seasons = ["24/25", "23/24", "22/23"]
solara_league = solara.reactive("EPL")
solara_season = solara.reactive("24/25")
solara_player = solara.reactive("Kevin De Bruyne")
solara_show_player = solara.reactive(False)
solara_players = solara.reactive([])
solara_player_features = solara.reactive(None)
analysis_results = solara.reactive("")

def predict(metric, features):
    if metric == "goals":
        return analysis_goals.predict(features).to_list()[0]
    if metric == "assists":
        return analysis_assists.predict(features).to_list()[0]
    if metric == "fouls":
        return analysis_fouls.predict(features).to_list()[0]

import solara
import json

# Reactive state variables
leagues = ['EPL', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
seasons = ["24/25", "23/24", "22/23"]
solara_league = solara.reactive("EPL")
solara_season = solara.reactive("24/25")
solara_player = solara.reactive("Kevin De Bruyne")
solara_show_player = solara.reactive(False)
solara_players = solara.reactive([])
solara_player_features = solara.reactive(None)
analysis_results = solara.reactive("")
solara_type = solara.reactive("")
solara_show_club = solara.reactive(False)
solara_club = solara.reactive("Arsenal")
solara_clubs = solara.reactive([])

def predict(metric, features):
    try:
        if metric == "goals":
            return analysis_goals.predict(features).to_list()[0]
        if metric == "assists":
            return analysis_assists.predict(features).to_list()[0]
        if metric == "fouls":
            return analysis_fouls.predict(features).to_list()[0]
    except IndexError:
        return 0

@solara.component
def PlayerAnalysis():
    """Displays player analysis results."""
    if solara_show_player.value:
        solara.Markdown(f"**Selected Player:** {solara_player.value}")
        
        if solara_player_features.value is not None:
            features = solara_player_features.value[solara_player_features.value['player'] == solara_player.value]
            # Simulate analysis
            goals = predict("goals", features)
            assists = predict("assists", features)
            fouls = predict("fouls", features)
            analysis_results = f"""Analysis Results: Goals: {goals} Assists: {assists} Fouls: {fouls} """
            real_value = f"""Real Value: Goals: {features['goals'].iloc[0]} Assists: {features['assists'].iloc[0]} Fouls: {features['fouls'].iloc[0]}"""
            if features['goals'].iloc[0] - goals > 1.5:
                solara.Markdown("""
                            <span style="color: red;">MORE GOALS THAN PREDICTOR</span>
                                """)
            else:
                solara.Markdown("""
                            <span style="color: green;">LESS GOALS THAN PREDICTOR</span>
                                """)
            if features['assists'].iloc[0] - assists > 1.5:
                solara.Markdown("""
                            <span style="color: red;">MORE ASSISTS THAN PREDICTOR</span>
                                """)
            else:
                solara.Markdown("""
                            <span style="color: green;">LESS ASSISTS THAN PREDICTOR</span>
                                """)
            if features['fouls'].iloc[0] - fouls >  1.5:
                solara.Markdown("""
                            <span style="color: red;">MORE FOULS THAN PREDICTOR</span>
                                """)
            else:
                solara.Markdown("""
                            <span style="color: green;">LESS FOULS THAN PREDICTOR</span>
                                """)
            
            solara.display(analysis_results)
            solara.display(real_value)

            solara.Markdown("### Based on goals")
            solara.DataFrame(club_analysis_goals.get_players_of_club(
                club_analysis_goals.get_club_of_players([solara_player.value])[0]
                ).sort_values(by='goals', ascending=False))
            solara.Markdown("### Based on assists")
            solara.DataFrame(club_analysis_assists.get_players_of_club(
                club_analysis_assists.get_club_of_players([solara_player.value])[0]
                ).sort_values(by='assists', ascending=False))
            solara.Markdown("### Based on fouls")
            solara.DataFrame(club_analysis_fouls.get_players_of_club(
                club_analysis_fouls.get_club_of_players([solara_player.value])[0]
                ).sort_values(by='fouls', ascending=False))

        else:
            solara.Markdown("No player features available.")
            
        solara.Markdown("**Similar Players:**")
        # Example JSON file handling for similar players
        try:
            with open('similar_player_goals.json', 'r', encoding='utf-8') as file:
                similar_goals = json.load(file)
                solara.Markdown("Similarity in terms of goals related features.")
                solara.display(similar_goals[solara_player.value])
        except FileNotFoundError:
            solara.Markdown("**Error:** Could not find similar players for goals.")
        try:
            with open('similar_player_assists.json', 'r', encoding='utf-8') as file:
                similar_assists = json.load(file)
                solara.Markdown("Similarity in terms of assists related features.")
                solara.display(similar_assists[solara_player.value])
        except FileNotFoundError:
            solara.Markdown("**Error:** Could not find similar players for assists.")
        try:
            with open('similar_player_fouls.json', 'r', encoding='utf-8') as file:
                similar_fouls = json.load(file)
                solara.Markdown("Similarity in terms of assists related features.")
                solara.display(similar_fouls[solara_player.value])
        except FileNotFoundError:
            solara.Markdown("**Error:** Could not find similar players for fouls.")  
        all_in_string = similar_fouls[solara_player.value] + similar_assists[solara_player.value] + similar_goals[solara_player.value]
        all_in_string = [x['player'] for x in all_in_string]
        solara.Markdown("String to Use in Comparator Page")
        solara.Markdown(",".join(all_in_string))


@solara.component
def ClubData():
    with solara.Card("Club Data"):
        solara.Markdown("Goals")
        solara.DataFrame(club_analysis_goals.get_players_of_club(solara_club.value).sort_values('goals', ascending=False)[['goals', 'player']])
        solara.Markdown("Assists")
        solara.DataFrame(club_analysis_assists.get_players_of_club(solara_club.value).sort_values('assists', ascending=False)[['assists', 'player']])
        solara.Markdown("Fouls")
        solara.DataFrame(club_analysis_fouls.get_players_of_club(solara_club.value).sort_values('fouls', ascending=False)[['fouls', 'player']])
        solara.Markdown("Was Fouled")
        solara.DataFrame(club_analysis_fouls.get_players_of_club(solara_club.value).sort_values('wasFouled', ascending=False)[['wasFouled', 'player']])
        solara.Markdown("Shots on Target")
        solara.DataFrame(club_analysis_fouls.get_players_of_club(solara_club.value).sort_values('shotsOnTarget', ascending=False)[['shotsOnTarget', 'player']])
        solara.Markdown("Total Shots")
        solara.DataFrame(club_analysis_fouls.get_players_of_club(solara_club.value).sort_values('totalShots', ascending=False)[['totalShots', 'player']])


@solara.component
def Page():
    solara.lab.ThemeToggle()
    solara.Title("Sleeping Bets")
    solara.Markdown("#😴 Sleeping Bets 😴")
    
    def fetch_data():
        # Update players and features
        data = ss.scrape_player_league_stats(solara_season.value, solara_league.value)
        solara_players.value = data['player'].to_list()
        solara_players.value.sort()
        solara_player_features.value = data
        solara_show_player.value = True
    
    def fetch_data_club():
        # Update players and features
        data = ss.scrape_player_league_stats(solara_season.value, solara_league.value)
        solara_clubs.value = list(set(data['team'].to_list()))
        solara_clubs.value.sort()
        solara_show_player.value = False
        solara_show_club.value = True
    
    with solara.Card("Player Section"):
        solara.Markdown("### Select League and Season")
        solara.Select(label="League", values=leagues, value=solara_league)
        solara.Select(label="Season", values=seasons, value=solara_season)
        solara.Select(label="Player/Club", values=["Player", "Club"], value=solara_type)
        if solara_type.value == "Player":
            solara.Button("Fetch Players", on_click=fetch_data)
        elif solara_type.value == "Club":
            solara.Button("Fetch Club", on_click=fetch_data_club)

        if solara_show_club.value:
            solara.Markdown("### Select Club")
            solara.Select(label="Club", values=solara_clubs.value, value=solara_club)
            solara.Button("Fetch Data", on_click=lambda: None)  # You can call analysis directly here
            ClubData()
    
        if solara_show_player.value:
            solara.Markdown("### Select Player")
            solara.Select(label="Player", values=solara_players.value, value=solara_player)
            solara.Button("Run Analysis", on_click=lambda: None)  # You can call analysis directly here
            PlayerAnalysis()

Page()