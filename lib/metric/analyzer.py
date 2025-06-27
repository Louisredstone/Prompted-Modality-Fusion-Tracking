import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import os

from ..data.data_resource import BaseImageSequenceDataResource, TrackResultDataResource, BaseImageSequence

def AUC(x, y, max_y=None) -> float:
    # self-defined AUC function (Area Under the Curve)
    x = np.array(x)
    y = np.array(y)
    assert x.ndim==1, "x should be a 1-D array"
    assert y.ndim in [1,2], "y should be a 1-D or 2-D array"
    assert x.shape[0]==y.shape[-1], "x and y should have the same length"
    idx = np.argsort(x)
    x = x[idx]
    y = y[:, idx] if y.ndim==2 else y[idx]
    axis = 1 if y.ndim==2 else 0
    if not max_y: max_y = np.max(y, axis=axis)
    if y.ndim==1:
        area = np.sum([max(y[i], y[i+1])*(x[i+1]-x[i]) for i in range(len(x)-1)])
        return area / (max_y*(x[-1]-x[0]))
    else: # y.ndim==2
        area = (np.max([y[:, :-1], y[:,1:]], axis=0)*(x[1:]-x[:-1])).sum(axis=1)
        return area / (max_y*(x[-1]-x[0]))

class Analyzer:
    # analyzer.metrics['center_error']: Center Error, Precision Rate, PR
    # analyzer.metrics['overlap_rate']: Overlap Rate, Success Rate, SR
    # analyzer.metrics['precision']: Precision, Pr
    # analyzer.metrics['recall']: Recall, Recall Rate, Re
    # analyzer.metrics['f_score']: F-score, F1-score, F-measure, F 
    
    def __init__(self, tracker_name: str, golden: BaseImageSequenceDataResource, track_result: TrackResultDataResource, work_dir: str): # root: root working directory
        self.debug=False
        self.tracker_name = tracker_name
        self.golden = golden
        self.track_result = track_result
        self.norm_dst = False # TODO: implement this
        self.work_dir = work_dir
        
        self.curves_of_sequences = {}
        self.thresholds = {}
        self.metrics = {}
        
        self.init_thresholds()
        self.init_curves_of_sequences()
        self.init_metrics()
        # self.run_vot_eval()

    @property
    def valid_metric_types(self) -> set[str]:
        return {'precision','recall', 'f_score', 'center_error', 'overlap_rate'}
    
    @property
    def metric_aliases(self) -> dict[str, list[str]]:
        return {
            'center_error': ['Center Error', 'Precision Rate', 'PR'],
            'overlap_rate': ['Overlap Rate', 'Success Rate', 'SR'],
            'precision': ['Precision', 'Pr'],
            'recall': ['Recall', 'Recall Rate', 'Re'],
            'f_score': ['F-score', 'F1-score', 'F-measure', 'F']
            }

    def init_thresholds(self):
        self.thresholds['overlap_rate'] = np.arange(0, 1.05, 0.05)
        self.thresholds['center_error'] = np.arange(0, 51)
        self.thresholds['precision'] = np.arange(0, 1.05, 0.05)
        self.thresholds['recall'] = np.arange(0, 1.05, 0.05)
        self.thresholds['f_score'] = np.arange(0, 1.05, 0.05)

    def init_curves_of_sequences(self):
        overlap_rate_thresholds = self.thresholds['overlap_rate']
        center_error_thresholds = self.thresholds['center_error']
        precision_thresholds = self.thresholds['precision']
        recall_thresholds = self.thresholds['recall']
        f_score_thresholds = self.thresholds['f_score']
        
        if self.norm_dst:
            center_error_thresholds = center_error_thresholds / 100

        overlap_rate_curve_of_sequences = np.zeros((len(self.golden), len(overlap_rate_thresholds)))
        center_error_curve_of_sequences = np.zeros((len(self.golden), len(center_error_thresholds)))
        precision_curve_of_sequences = np.zeros((len(self.golden), len(precision_thresholds)))
        recall_curve_of_sequences = np.zeros((len(self.golden), len(recall_thresholds)))
        f_score_curve_of_sequences = np.zeros((len(self.golden), len(f_score_thresholds)))

        for i_seq, seq_golden in enumerate(self.golden):  # for each sequence
            seq_golden: BaseImageSequence
            bboxes_ltwh_golden = seq_golden.bboxes_ltwh
            width, height = seq_golden.width, seq_golden.height

            seq_pred = self.track_result.getSeqByName(seq_golden.name)
            if seq_pred is None:
                print(f"Track result of seq '{seq_golden.name}' not found, skipping...")
                continue
            print(f'evaluating {self.tracker_name} on {seq_golden.name} ...')

            bboxes_ltwh_pred = seq_pred.bboxes_ltwh
            if bboxes_ltwh_pred.size == 0:
                print(f"No detection result for seq '{seq_golden.name}', skipping...")
                continue

            seq_length = len(seq_golden)

            if bboxes_ltwh_pred.shape[0] != seq_length:
                bboxes_ltwh_pred = bboxes_ltwh_pred[:seq_length, :]

            def find_first_true_index(arr)-> int: return int(np.argmax(arr))
            first_valid_index = find_first_true_index(seq_golden.valid)
            # first_valid_index: usually 0, but sometimes > 0.

            for i_frame in range(first_valid_index + 1, seq_length):
                # Reader memo: 此处从1开始, 因为第0帧的检测结果是由GT初始化的, 所以不参与评估.
                bbox_pred = bboxes_ltwh_pred[i_frame, :]
                bbox_golden = bboxes_ltwh_golden[i_frame, :]
                if (np.isnan(bbox_pred).any() or not np.isreal(bbox_pred).all() or bbox_pred[2] <= 0 or bbox_pred[3] <= 0) and not np.isnan(bbox_golden).any():
                    bboxes_ltwh_pred[i_frame, :] = bboxes_ltwh_pred[i_frame-1, :]
                    # 这里的处理方式是，如果检测结果不合法（比如检测框的宽或高为0），则用上一帧的检测结果代替. 但这可能引发问题, 因为有时丢失目标是正常的, 目标物体可能受到遮挡. 因此, 这里需要进一步讨论.
                    # 此外, 有些sequence从第一帧开始是invalid, 因此需要跳过这些帧.

            bboxes_ltwh_pred[:first_valid_index+1, :] = bboxes_ltwh_golden[:first_valid_index+1, :]

            center_gt = np.column_stack((bboxes_ltwh_golden[:, 0] + (bboxes_ltwh_golden[:, 2] - 1) / 2,
                                        bboxes_ltwh_golden[:, 1] + (bboxes_ltwh_golden[:, 3] - 1) / 2))
            # Reader memo: groundtruth框中心坐标

            center_pred = np.column_stack((bboxes_ltwh_pred[:, 0] + (bboxes_ltwh_pred[:, 2] - 1) / 2,
                                    bboxes_ltwh_pred[:, 1] + (bboxes_ltwh_pred[:, 3] - 1) / 2))
            # Reader memo: tracker框中心坐标

            if self.norm_dst:
                center_pred[:, 0] /= bboxes_ltwh_golden[:, 2]
                center_pred[:, 1] /= bboxes_ltwh_golden[:, 3]
                center_gt[:, 0] /= bboxes_ltwh_golden[:, 2]
                center_gt[:, 1] /= bboxes_ltwh_golden[:, 3]

            center_error = np.sqrt(np.sum((center_pred - center_gt) ** 2, axis=1))

            # index = anno > 0
            # idx = np.all(index, axis=1)
            idx = np.all(bboxes_ltwh_golden > 0, axis=1)
            # Reader memo: 这里的idx是指标注矩形框中有目标的帧, 即idx=True的位置.
            # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.
            # 这回答了之前关于丢失目标的疑问.
            
            bboxes_pred = bboxes_ltwh_pred.copy()
            bboxes_pred[:, 2] += bboxes_pred[:, 0]
            bboxes_pred[:, 3] += bboxes_pred[:, 1]
            bboxes_gt = bboxes_ltwh_golden.copy()
            bboxes_gt[:, 2] += bboxes_gt[:, 0]
            bboxes_gt[:, 3] += bboxes_gt[:, 1]
            
            area_overlap = (np.maximum(
                                0, 
                                np.minimum(bboxes_pred[:, 2], bboxes_gt[:, 2]) - np.maximum(bboxes_pred[:, 0], bboxes_gt[:, 0])
                                # min(right_pred, right_gt) - max(left_pred, left_gt)
                            )) * (np.maximum(
                                0, 
                                np.minimum(bboxes_pred[:, 3], bboxes_gt[:, 3]) - np.maximum(bboxes_pred[:, 1], bboxes_gt[:, 1])
                                # min(bottom_pred, bottom_gt) - max(top_pred, top_gt)
                            ))
            area_pred = bboxes_ltwh_pred[:, 2] * bboxes_ltwh_pred[:, 3]
            area_gt = bboxes_ltwh_golden[:, 2] * bboxes_ltwh_golden[:, 3]
            area_all = height * width # scalar
            
            overlap_rate = area_overlap / (area_pred + area_gt - area_overlap)
            overlap_rate[~idx] = -1
            center_error[~idx] = -1

            # calculate confusion matrix
            true_positive = area_overlap
            false_positive = area_pred - area_overlap
            false_negative = area_gt - area_overlap
            true_negative = area_all - area_pred - area_gt + area_overlap
            true_positive_rate = true_positive / (true_positive + false_negative) # might encounter zero division
            false_positive_rate = false_positive / (false_positive + true_negative) # might encounter zero division
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f_score = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

            for t_idx, threshold in enumerate(overlap_rate_thresholds):
                overlap_rate_curve_of_sequences[i_seq, t_idx] = np.sum(overlap_rate > threshold) / len(bboxes_ltwh_golden)

            for t_idx, threshold in enumerate(center_error_thresholds):
                center_error_curve_of_sequences[i_seq, t_idx] = np.sum(center_error <= threshold) / len(bboxes_ltwh_golden)
            
            for t_idx, threshold in enumerate(precision_thresholds):
                precision_curve_of_sequences[i_seq, t_idx] = np.sum(precision > threshold) / len(bboxes_ltwh_golden)
                
            for t_idx, threshold in enumerate(recall_thresholds):
                recall_curve_of_sequences[i_seq, t_idx] = np.sum(recall > threshold) / len(bboxes_ltwh_golden)
                
            for t_idx, threshold in enumerate(f_score_thresholds):
                f_score_curve_of_sequences[i_seq, t_idx] = np.sum(f_score > threshold) / len(bboxes_ltwh_golden)

        # if not os.path.exists(tmp_mat_path):
        #     os.makedirs(tmp_mat_path)

        # # dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_overlap_{eval_type}.npz')
        # dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_alg_overlap_rate.npz')
        # np.savez(dataName1, ave_success_rate_plot=overlap_rate_curve_of_sequences, tracker=tracker)

        # # dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_error_{eval_type}.npz')
        # dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_alg_center_error.npz')
        # np.savez(dataName2, ave_success_rate_plot=center_error_curve_of_sequences, tracker=tracker)
        
        # return overlap_rate_curve_of_sequences, center_error_curve_of_sequences, precision_curve_of_sequences, recall_curve_of_sequences, f_score_curve_of_sequences
        self.curves_of_sequences['overlap_rate'] = overlap_rate_curve_of_sequences
        self.curves_of_sequences['center_error'] = center_error_curve_of_sequences
        self.curves_of_sequences['precision'] = precision_curve_of_sequences
        self.curves_of_sequences['recall'] = recall_curve_of_sequences
        self.curves_of_sequences['f_score'] = f_score_curve_of_sequences
        
    def init_metrics(self):
        self.metrics['center_error'] = float(AUC(self.thresholds['center_error'], self.curves_of_sequences['center_error'], max_y=1).mean())
        self.metrics['overlap_rate'] = float(AUC(self.thresholds['overlap_rate'], self.curves_of_sequences['overlap_rate'], max_y=1).mean())
        self.metrics['precision'] = float(AUC(self.thresholds['precision'], self.curves_of_sequences['precision'], max_y=1).mean())
        self.metrics['recall'] = float(AUC(self.thresholds['recall'], self.curves_of_sequences['recall'], max_y=1).mean())
        self.metrics['f_score'] = float(AUC(self.thresholds['f_score'], self.curves_of_sequences['f_score'], max_y=1).mean())
        
    def run_vot_eval(self):
        # FUTURE WARNING: This function is a temporary solution, and will be replaced by a more elegant implementation.
        vot_workspace = os.path.join(self.work_dir, 'vot_workspace')
        if not os.path.exists(vot_workspace):
            os.makedirs(vot_workspace)
        from vot import config

        from vot.analysis import AnalysisProcessor
        from vot.report import generate_serialized
        # from vot.workspace import Workspace
        from vot.workspace.storage import Cache

        # workspace = Workspace.load(args.workspace)
        # Nope. We need to define our own workspace here.

        # logger.debug("Loaded workspace in '%s'", args.workspace)

        # if not args.trackers:
        #     trackers = workspace.list_results(workspace.registry)
        # else:
        #     trackers = workspace.registry.resolve(*args.trackers, storage=workspace.storage.substorage("results"), skip_unknown=False)
        from vot.tracker import Tracker
        trackers = [Tracker(self.tracker_name, '', self.tracker_name)]

        if not trackers:
            logger.warning("No trackers resolved, stopping.")
            return

        logger.debug("Running analysis for %d trackers", len(trackers))

        if config.worker_pool_size == 1:

            if self.debug:
                from vot.analysis.processor import DebugExecutor
                logging.getLogger("concurrent.futures").setLevel(logging.DEBUG)
                executor = DebugExecutor()
            else:
                from vot.utilities import ThreadPoolExecutor
                executor = ThreadPoolExecutor(1)

        else:
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(config.worker_pool_size)

        if not config.persistent_cache:
            from cachetools import LRUCache
            cache = LRUCache(1000)
        else:
            from vot.workspace.storage import Proxy, LocalStorage, NullStorage
            storage = Proxy(lambda: LocalStorage(self.work_dir) if self.work_dir is not None else NullStorage())
            cache = Cache(storage.substorage("cache").substorage("analysis"))

        from threading import RLock, Condition
        from concurrent.futures import Executor, Future, ThreadPoolExecutor
        from vot.utilities import class_fullname, Progress
        from vot.workspace import StackLoader

        def my_process_stack_analyses(experiments: list[Experiment], sequences, trackers: list[Tracker]):
            """modified from vot.analysis.processor.process_stack_analyses"""
            processor = AnalysisProcessor.default()

            results = dict()
            condition = Condition()
            errors = []

            def insert_result(container: dict, key):
                """Creates a callback that inserts the result of a computation into a container. The container is a dictionary that maps analyses to their results.
                
                Args:
                    container (dict): The container to insert the result into.
                    key (Analysis): The analysis to insert the result for.
                """
                def insert(future: Future):
                    """Inserts the result of a computation into a container."""
                    try:
                        container[key] = future.result()
                    except Exception as e:
                        errors.append(e)
                    with condition:
                        condition.notify()
                return insert

            if isinstance(trackers, Tracker): trackers = [trackers]

            for experiment in experiments: # TODO: prepare stack or experiments

                logger.debug("Traversing experiment %s", experiment.identifier)

                experiment_results = dict()

                results[experiment] = experiment_results

                # sequences = experiment.transform(workspace.dataset) # TODO

                for analysis in experiment.analyses:

                    if not analysis.compatible(experiment):
                        continue

                    logger.debug("Traversing analysis %s", class_fullname(analysis))

                    with condition:
                        experiment_results[analysis] = None
                    promise = processor.commit(analysis, experiment, trackers, sequences)
                    promise.add_done_callback(insert_result(experiment_results, analysis))

            if processor.total == 0:
                return results

            logger.debug("Waiting for %d analysis tasks to finish", processor.total)

            with Progress("Running analysis", processor.total) as progress:
                try:

                    while True:

                        progress.absolute(processor.total - processor.pending)
                        if processor.pending == 0:
                            progress.absolute(processor.total)
                            break

                        with condition:
                            condition.wait(1)

                except KeyboardInterrupt:
                    processor.cancel()
                    progress.close()
                    logger.info("Analysis interrupted by user, aborting.")
                    return None

            if len(errors) > 0:
                logger.info("Errors occured during analysis, incomplete.")
                for e in errors:
                    logger.info("Failed task {}: {}".format(e.task, e.root_cause))
                    #if logger.isEnabledFor(logging.DEBUG):
                    #    e.print(logger)
                return None

            return results

        try:

            with AnalysisProcessor(executor, cache):

                # ORIGIN: results = process_stack_analyses(workspace, trackers)
                results = my_process_stack_analyses(experiments, sequence, trackers)

                if results is None:
                    return

                if args.name is None:
                    name = "{:%Y-%m-%dT%H-%M-%S.%f%z}".format(datetime.now())
                else:
                    name = args.name

                storage = workspace.storage.substorage("analysis")

                if args.format == "json":
                    generate_serialized(trackers, workspace.dataset, results, storage, "json", name)
                elif args.format == "yaml":
                    generate_serialized(trackers, workspace.dataset, results, storage, "yaml", name)
                else:
                    raise ValueError("Unknown format '{}'".format(args.format))

                logger.info("Analysis successful, report available as %s", name)

        finally:

            executor.shutdown(wait=True)