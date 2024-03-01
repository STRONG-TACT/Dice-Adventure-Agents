import os
from collections import Counter
from collections import defaultdict
from datetime import datetime
from dateutil.parser import parse
from os import makedirs
from os import path
from time import sleep
from time import time
import tensorflow as tf
from threading import Thread


class GameMetricsTracker:
    def __init__(self, level, metrics_config, instance_id=1, model_number=1):
        self.id = instance_id
        self.model_number = model_number
        self.level = level
        self.save_threshold = 10000
        self.num_records = 0
        self.metrics_config = metrics_config
        # Init values
        self.repeat_counter = Counter()
        self.num_agent_actions = 0
        self.num_rounds = 0
        self.num_games = 0
        self.num_phases = 0
        self.num_team_deaths = 0
        self.clock_start = self._timestamp(as_string=False)
        self.level_start = self._timestamp(as_string=False)
        # Time Series Lists
        self.agent_actions = defaultdict(list)
        self.actions = Counter()
        self.levels = defaultdict(list)
        self.team_deaths = []
        self.rounds = []
        self.games = []
        self.phases = []
        self.player_trackers = {"Dwarf": PlayerMetricsTracker("Dwarf"),
                                "Giant": PlayerMetricsTracker("Giant"),
                                "Human": PlayerMetricsTracker("Human")}
        # Time series counter
        self.metric_counter = Counter()
        self.metrics_dir = self.metrics_config["DIRECTORIES"]["LOGFILES"].format(self.model_number)
        self.tb_dir = self.metrics_config["DIRECTORIES"]["TENSORBOARD"]
        self._setup_directories()

        # Start tensorboard writer
        TensorBoardWriter(metrics_config=self.metrics_config, model_number=self.model_number)

    def save(self):
        ##############
        # GAME LEVEL #
        ##############
        # Levels
        self._save_level_metrics()

        self._reset()

    def _setup_directories(self):
        makedirs(self.metrics_dir, exist_ok=True)
        for metric in self.metrics_config["GAME"]:
            makedirs(self.metrics_dir + self.metrics_config["GAME"][metric]["SUBDIRECTORY"], exist_ok=True)

    def _get_filepath(self, component, metric, params=None, additional_info=""):
        if params is None:
            params = []
        if not isinstance(params, list):
            params = [params]
        return self.metrics_dir + self.metrics_config[component][metric]["SUBDIRECTORY"] + \
            self.metrics_config[component][metric]["FILENAME"].format(*params) + additional_info + \
            self.metrics_config["EXTENSION"].format(self.id)

    def _save_level_metrics(self, component="GAME", metric="LEVEL"):
        for level in self.levels:
            graph_name = self.metrics_config[component][metric]["GRAPH_NAME"].format(level)
            filepath = self._get_filepath(component, metric, params=level, additional_info="-gn-{}".format(graph_name))
            columns = self.metrics_config[component][metric]["COLUMNS"]
            self._save_records(records=self.levels[level], columns=columns, filepath=filepath)

    @staticmethod
    def _save_records(records, columns, filepath):
        mode = "a" if path.exists(filepath) else "w"
        with open(filepath, mode) as file:
            # Need to write columns if first time
            if mode == "w":
                file.write("\t".join(columns)+"\n")
            for rec in records:
                # self.metric_counter[metric_counter_name] += 1

                file.write("\t".join([str(i) for i in rec])+"\n")

    def _reset(self):
        self.levels = defaultdict(list)

    def update(self, target, **kwargs):
        self.num_records += 1
        if target == "game":
            self._update_game(**kwargs)
        else:
            self._update_player(**kwargs)
        if self.num_records >= self.save_threshold:
            self.num_records = 0
            self.save()

    def _update_player(self, player, metric_name, pin_type=None,
                       combat_outcome=None, enemy_type=None, enemy_size=None):
        timestamp = self._timestamp()

        if metric_name == "pins":
            self.player_trackers[player].pin(pin_type, timestamp)
        elif metric_name == "combat":
            self.player_trackers[player].combat(enemy_type, enemy_size,
                                                combat_outcome, timestamp)
        else:
            self.player_trackers[player].generics(metric_name, timestamp)

    def _update_game(self, metric_name, player=None, agent_action=None, level=None, phase=None):
        timestamp = self._timestamp()

        if metric_name == "new_phase":
            self.num_phases += 1
            self.phases.append([timestamp, phase, self._calculate_time_elapsed(self.phases)])

        elif metric_name == "new_round":
            self.num_rounds += 1
            self.rounds.append([timestamp, self.num_rounds, self._calculate_time_elapsed(self.rounds)])

        elif metric_name == "game_over":
            self.num_games += 1
            self.games.append([timestamp, self.num_games, self._calculate_time_elapsed(self.games)])
            self.save()
        elif metric_name == "new_level":
            # print("LEVEL HAS CHANGED!")
            # print(F"level is: {self.level}")
            # Track time to complete last level
            elapsed_time = (self._timestamp(as_string=False) - self.level_start)
            self.levels[self.level].append([timestamp, self.level, self.repeat_counter[self.level],
                                            elapsed_time.total_seconds()])
            # Update level
            self.level = level
            self.level_start = self._timestamp(as_string=False)
            self.repeat_counter[self.level] += 1
        elif metric_name == "num_repeats":
            # print("LEVEL STAYED THE SAME! ", level)
            # Track time to complete last level
            elapsed_time = (self._timestamp(as_string=False) - self.level_start)
            self.levels[self.level].append([timestamp, self.level, self.repeat_counter[self.level],
                                            elapsed_time.total_seconds()])

            self.repeat_counter[self.level] += 1
            self.level_start = self._timestamp(as_string=False)
            # self.save()
        elif metric_name == "agent_action":
            self.num_agent_actions += 1
            self.agent_actions[player].append([timestamp, self.level, self.num_rounds, phase, agent_action])
            self.actions[agent_action] += 1
            # print(self.actions)
        elif metric_name == "team_death":
            self.num_team_deaths += 1
            self.team_deaths.append([timestamp, self.level, self.num_rounds])

    # new round vs new phase
    def _calculate_time_elapsed(self, time_series):
        if time_series:
            # Get time elapsed since last round
            time_elapsed = self._timestamp(as_string=False) - parse(time_series[-1][0])
        else:
            time_elapsed = self._timestamp(as_string=False) - self.clock_start
        return time_elapsed.seconds

    @staticmethod
    def _timestamp(as_string=True):
        timestamp = datetime.utcnow()
        if as_string:
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')

        return timestamp


class PlayerMetricsTracker:
    def __init__(self, player):
        self.player = player
        self.health_loss = []
        self.deaths = []
        self.pins = {"pinga": [], "pingb": [], "pingc": [], "pingd": []}
        # Combat Tracking
        self.total_wins = 0
        self.total_losses = 0
        self.wins = self._get_enemy_tracker()
        self.losses = self._get_enemy_tracker()

    def pin(self, pin_type, timestamp):
        self.pins[pin_type].append(timestamp)

    def combat(self, enemy_type, enemy_size, outcome, timestamp):
        if outcome == "win":
            self.total_wins += 1
            self.wins[enemy_type][enemy_size].append(timestamp)
        else:
            self.total_losses += 1
            self.losses[enemy_type][enemy_size].append(timestamp)

    def generics(self, metric_name, timestamp):
        if metric_name == "death":
            self.deaths.append(timestamp)
        elif metric_name == "health_loss":
            self.health_loss.append(timestamp)

    @staticmethod
    def _get_enemy_tracker():
        return {"Monster": {"S": [], "M": [], "L": [], "XL": []},
                "Stone": {"S": [], "M": [], "L": []},
                "Trap": {"S": [], "M": [], "L": []}}


class TensorBoardWriter:
    def __init__(self, metrics_config, model_number):
        self.metrics_config = metrics_config
        self.model_number = model_number

        self.metrics_dir = self.metrics_config["DIRECTORIES"]["LOGFILES"].format(self.model_number)
        self.tb_dir = self.metrics_config["DIRECTORIES"]["TENSORBOARD"].format(self.model_number)
        # TB Logger
        # self.tf_writer = tf.summary.create_file_writer(self.tb_dir)
        # self.metric_counter = Counter()
        # self.record_offset = Counter()

        logging_thread = Thread(target=self.logger, args=(self.metrics_dir, self.tb_dir,
                                                          self.metrics_config["TB_LOGGER_REFRESH_RATE"]))
        logging_thread.start()

    @staticmethod
    def logger(metrics_dir, tb_dir, refresh_rate):
        tf_writer = tf.summary.create_file_writer(tb_dir, flush_millis=30000)
        metric_counter = Counter()

        dirs = [metrics_dir + dir_ for dir_ in os.listdir(metrics_dir)]
        with tf_writer.as_default():
            while True:
                sleep(refresh_rate)
                for d in dirs:
                    files = [d + "/" + file for file in os.listdir(d)]
                    for file in files:
                        graph_name = file.split("-")[-2]
                        num_cols = None
                        with open(file, "r") as logfile:
                            c = 0
                            for line in logfile:
                                # Don't reread records already written to tensorboard
                                if c < metric_counter[graph_name]:
                                    c += 1
                                    continue

                                line = line.split("\t")
                                if not num_cols:
                                    num_cols = len(line)
                                # This is an incomplete line, so finish reading file
                                if len(line) < num_cols:
                                    break
                                if c > 0:
                                    tf.summary.scalar(name="{}/{}".format(d.split("/")[-1], graph_name),
                                                      data=float(line[-1]),
                                                      step=metric_counter[graph_name])
                                    metric_counter[graph_name] += 1
                                c += 1
                        tf_writer.flush()








