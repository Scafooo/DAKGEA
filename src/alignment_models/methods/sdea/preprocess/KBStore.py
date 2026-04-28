import re
from collections import defaultdict

from tqdm import tqdm

from .._globals import args, functionality_control, functionality_threshold
from . import Parser
from .Parser import OEAFileType
from ..tools import FileTools
from ..tools.Announce import Announce
from ..tools.MultiprocessingTool import MPTool, MultiprocessingTool
from ..tools.MyTimer import MyTimer
from ..tools.text_to_word_sequence import text_to_word_sequence


class KBStore:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.entities = []
        self.literals = []
        self.entity_ids = {}
        self.classes_ids = {}
        self.literal_ids = {}

        self.relations = []
        self.properties = []
        self.relation_ids = {}
        self.property_ids = {}

        self.words = []
        self.word_ids = {}

        self.facts = {}
        self.literal_facts = {}
        self.blocks = {}
        self.word_level_blocks = {}

        self.properties_functionality = None
        self.relations_functionality = None

    def load_kb(self) -> None:
        timer = MyTimer()
        self.load_path(self.dataset.attr, self.load, OEAFileType.attr)
        if args.relation:
            self.load_path(self.dataset.rel, self.load, OEAFileType.rel)
            self.relations_functionality = KBStore.calculate_func(
                self.relations, self.relation_ids, self.facts, self.entity_ids
            )
        for ent, facts in self.facts.items():
            facts.sort(key=lambda x: (x[0], x[1]), reverse=False)
        self.properties_functionality = KBStore.calculate_func(
            self.properties, self.property_ids, self.literal_facts, self.entity_ids
        )
        timer.stop()
        print(Announce.printMessage(), 'Finished loading in', timer.total_time())

    @staticmethod
    def load_path(path, load_func, file_type: OEAFileType) -> None:
        import os
        print(Announce.doing(), 'Start loading', path)
        if os.path.isdir(path):
            for file in sorted(os.listdir(path)):
                if os.path.isdir(file):
                    continue
                file = os.path.join(path, file)
                KBStore.load_path(file, load_func, file_type)
        else:
            load_func(path, file_type)
        print(Announce.done(), 'Finish loading', path)

    def load(self, file: str, file_type: OEAFileType) -> None:
        tuples = Parser.for_file(file, file_type)
        with tqdm(desc='add tuples') as tqdm_add:
            tqdm_add.total = len(tuples)
            for args_t in tuples:
                self.add_tuple(*args_t, file_type)
                tqdm_add.update()

    def add_tuple(self, sbj: str, pred: str, obj: str, file_type: OEAFileType) -> None:
        assert sbj is not None and obj is not None and pred is not None
        if file_type == OEAFileType.attr:
            if obj.startswith('"'):
                obj = obj[1:-1]
            toks = text_to_word_sequence(obj)
            for tok in toks:
                if len(tok) < 5:
                    continue
                if bool(re.search(r'\d', tok)):
                    return
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)
            obj_id = self.get_or_add_item(obj, self.literals, self.literal_ids)
            pred_id = self.get_or_add_item(pred, self.properties, self.property_ids)
            self.add_fact(sbj_id, pred_id, obj_id, self.literal_facts)
            self.add_to_blocks(sbj_id, obj_id)
            words = text_to_word_sequence(obj)
            self.add_word_level_blocks(sbj_id, words)
        elif file_type == OEAFileType.rel:
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)
            obj_id = self.get_or_add_item(obj, self.entities, self.entity_ids)
            pred_id = self.get_or_add_item(pred, self.relations, self.relation_ids)
            pred2_id = self.get_or_add_item(pred + '-', self.relations, self.relation_ids)
            self.add_fact(sbj_id, pred_id, obj_id, self.facts)
            self.add_fact(obj_id, pred2_id, sbj_id, self.facts)

    def add_item(self, name: str, names: list, ids: dict) -> int:
        iid = len(names)
        names.append(name)
        ids[name] = iid
        return iid

    def get_or_add_item(self, name: str, names: list, ids: dict) -> int:
        if name in ids:
            return ids.get(name)
        else:
            return self.add_item(name, names, ids)

    def add_fact(self, sbj_id, pred_id, obj_id, facts_list: dict) -> None:
        if sbj_id in facts_list:
            facts_list[sbj_id].append((pred_id, obj_id))
        else:
            facts_list[sbj_id] = [(pred_id, obj_id)]

    def add_to_blocks(self, sbj_id, obj_id) -> None:
        if obj_id in self.blocks:
            self.blocks[obj_id].add(sbj_id)
        else:
            self.blocks[obj_id] = {sbj_id}

    def add_word_level_blocks(self, entity_id, words):
        for word in words:
            if word in self.word_level_blocks:
                self.word_level_blocks[word].add(entity_id)
            else:
                self.word_level_blocks[word] = {entity_id}

    @staticmethod
    def calculate_func(r_names: list, r_ids: dict, facts_list: dict, sbj_ids: dict) -> list:
        num_occurrences = [0] * len(r_names)
        func = [0.] * len(r_names)
        num_subjects_per_relation = [0] * len(r_names)
        last_subject = [-1] * len(r_names)
        for sbj_id in sbj_ids.values():
            facts = facts_list.get(sbj_id)
            if facts is None:
                continue
            for fact in facts:
                num_occurrences[fact[0]] += 1
                if last_subject[fact[0]] != sbj_id:
                    last_subject[fact[0]] = sbj_id
                    num_subjects_per_relation[fact[0]] += 1
        for r_name, rid in r_ids.items():
            if num_occurrences[rid] > 0:
                func[rid] = num_subjects_per_relation[rid] / num_occurrences[rid]
        return func
