import sys, os, logging
import tensorflow as tf
import numpy as np
from collections import defaultdict

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Trainer(object):
    def __init__(self):
        pass

    @staticmethod
    def _train_sess(model, batch_generator, steps, summary_writer, save_summary_steps):
        global_step = tf.train.get_or_create_global_step()

        for i in range(steps):
            train_batch = batch_generator.next()
            feed_dict = {ph: train_batch[key] for key, ph in model.input_placeholder_dict.items() if key in train_batch}
            # TODO Summary
            if i % save_summary_steps == 0:
                _, _, loss_val, summ, global_step_val = model.session.run([model.train_op, model.train_update_metrics, #TODO 
                                                                           model.total_loss, model.summary_op, global_step], #TODO
                                                                          feed_dict=feed_dict)
                if summary_writer is not None:
                    summary_writer.add_summary(summ, global_step_val)
            else:
                _, _, loss_val = model.session.run([model.train_op, model.train_update_metrics, model.total_loss], feed_dict=feed_dict)
            if np.isnan(loss_val):
                raise ValueError("NaN loss!")

        metrics_values = {k: v[0] for k, v in model.train_metrics.items()} # TODO
        metrics_val = model.session.run(metrics_values) # TODO
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logger.info("- Train metrics: " + metrics_string)

    @staticmethod
    def _eval_sess(model, batch_generator, steps, summary_writer):
        global_step = tf.train.get_or_create_global_step()

        final_output = [] # defaultdict(list)
        for _ in range(steps):
            eval_batch = batch_generator.next()
            eval_batch["training"] = False # TODO
            feed_dict = {ph: eval_batch[key] for key, ph in model.input_placeholder_dict.items() if key in eval_batch}
            _, output = model.session.run([model.eval_update_metrics, model.y_pred], feed_dict=feed_dict)
            [final_output.append(tmp) for tmp in output]

            # for key in output.keys():
            #     final_output[key] += [v for v in output[key]]

        # Get the values of the metrics
        # metrics_values = {k: v[0] for k, v in model.eval_metrics.items()} # TODO
        # metrics_val = model.session.run(metrics_values) # TODO
        # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        # logger.info("- Eval metrics: " + metrics_string)

        # Add summaries manually to writer at global_step_val
        # if summary_writer is not None:
        #     global_step_val = model.session.run(global_step)
        #     for tag, val in metrics_val.items():
        #         summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
        #         summary_writer.add_summary(summ, global_step_val)

        final_output = np.asarray(final_output)
        return final_output

    @staticmethod
    def _train_and_evaluate(model, train_batch_generator, eval_batch_generator, evaluator, epochs=1, eposides=1,
                            save_dir=None, summary_dir=None, save_summary_steps=1000):
        best_saver = tf.train.Saver(max_to_keep=1) if save_dir is not None else None
        train_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'train_summaries')) if summary_dir else None
        eval_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'eval_summaries')) if summary_dir else None

        best_eval_score = 0.0
        for epoch in range(epochs):
            logger.info("Epoch {}/{}".format(epoch + 1, epochs))
            train_batch_generator.init()
            train_num_steps = (train_batch_generator.get_instance_size() + train_batch_generator.get_batch_size() - 1) // train_batch_generator.get_batch_size()
            model.session.run(model.train_metric_init_op)

            # one epoch consists of several eposides
            assert isinstance(eposides, int)
            num_steps_per_eposide = (train_num_steps + eposides - 1) // eposides # 7//2 = 3
            for eposide in range(eposides):
                logger.info("Eposide {}/{}".format(eposide + 1, eposides))
                current_step_num = min(num_steps_per_eposide, train_num_steps - eposide * num_steps_per_eposide)
                eposide_id = epoch * eposides + eposide + 1
                Trainer._train_sess(model, train_batch_generator, current_step_num, train_summary, save_summary_steps)

                # # Save weights
                # if save_dir is not None:
                #     last_save_path = os.path.join(save_dir, 'last_weights', 'after-eposide')
                #     if os.path.exists(last_save_path) is False:
                #         os.makedirs(last_save_path)
                #     model.save(last_save_path, global_step=eposide_id)

                # Evaluate for one epoch on dev set
                eval_batch_generator.init()
                eval_instances = eval_batch_generator.get_instances()
                model.session.run(model.eval_metric_init_op)

                eval_num_steps = (eval_batch_generator.get_instance_size() + eval_batch_generator.get_batch_size() - 1) // eval_batch_generator.get_batch_size()
                output = Trainer._eval_sess(model, eval_batch_generator, eval_num_steps, eval_summary)
                output = output[:, 1] # 取 output[:, 1]
                score = evaluator.get_score(output, eval_instances) # TODO
                metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
                logger.info("- Eval metrics: " + metrics_string)
                
                # Save best weights
                eval_score = score[evaluator.get_monitor()] # TODO
                if eval_score > best_eval_score:
                    logger.info("- epoch %d eposide %d: Found new best score: %f" % (epoch + 1, eposide + 1, eval_score))
                    best_eval_score = eval_score
                    # Save best weights
                    if save_dir is not None:
                        tmp_path = os.path.join(save_dir, 'best_weights', 'after-eposide')
                        if os.path.exists(tmp_path) is False:
                            os.makedirs(tmp_path)
                        best_save_path = best_saver.save(model.session, tmp_path, global_step=eposide_id)
                        logger.info("- Found new best model, saving in {}".format(best_save_path))


    @staticmethod
    def _evaluate(model, batch_generator, evaluator):
        """ 模型中调用
        """
        pass
        # batch_generator.init()
        # eval_instances = batch_generator.get_instances()
        # eval_num_steps = (len(eval_instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        # output = Trainer._eval_sess(model, batch_generator, eval_num_steps, None) # TODO
        # pred_answer = model.get_best_answer(output, eval_instances) # TODO
        # score = evaluator.get_score(pred_answer) # TODO
        # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
        # logger.info("- Eval metrics: " + metrics_string)

    @staticmethod
    def _inference(model, batch_generator):
        """ 模型中调用
        """
        batch_generator.init()
        model.session.run(model.eval_metric_init_op)
        instances = batch_generator.get_instances()
        eval_num_steps = (len(instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer.batch_inference(model, batch_generator, eval_num_steps)
        return output
    
    @staticmethod
    def _single_inference(model, batch_generator):
        """ 模型中调用
        """
        batch_generator.init()
        model.session.run(model.eval_metric_init_op)
        instances = batch_generator.get_instances()
        eval_num_steps = (len(instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer.batch_inference(model, batch_generator, eval_num_steps)
        return output

    @staticmethod
    def batch_inference(model, batch_generator, steps):
        """ 当前文件中使用
        """
        # global_step = tf.train.get_or_create_global_step()
        final_output = []
        for _ in range(steps):
            eval_batch = batch_generator.next()
            eval_batch["training"] = False
            feed_dict = {ph: eval_batch[key] for key, ph in model.input_placeholder_dict.items() if key in eval_batch and key not in ['answer_start','answer_end','is_impossible']}
            output = model.session.run(model.y_pred, feed_dict=feed_dict)
            for tmp in output:
                final_output.append(tmp)
        return final_output
