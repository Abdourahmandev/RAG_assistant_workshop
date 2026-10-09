from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from rag.application import Progress

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_prepare_then_chat_reruns_do_not_reindex_and_keep_history(self):
        fake = SimpleNamespace(cached_passages=0, close=Mock(), answer=Mock(return_value={
            "answer": "Une réponse de test [a-6_002_p0004_0195].",
            "passages": [{"id": "a-6_002_p0004_0195", "title": "PDF, page 4", "text": "Une preuve."}],
            "total_s": .2,
        }))

        def prepare(plan, **kwargs):
            kwargs["progress"](Progress("Complete", len(plan.chunks), len(plan.chunks)))
            return fake

        with patch("rag.application.prepare_assistant", side_effect=prepare) as prepare_mock:
            ui = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            self.assertFalse(ui.exception)
            self.assertTrue(ui.chat_input[0].disabled)
            prepare_mock.assert_not_called()
            ui.button(key="prepare").click().run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.chat_input[0].disabled)
            ui.chat_input[0].set_value("Une première question ?").run()
            self.assertFalse(ui.exception)
            self.assertEqual(len(ui.chat_message), 2)
            self.assertEqual(fake.answer.call_count, 1)
            self.assertFalse(ui.button(key="clear_chat").disabled)
            ui.chat_input[0].set_value("Et ensuite ?").run()
            self.assertFalse(ui.exception)
            self.assertEqual(len(fake.answer.call_args.args[1]), 2)
            self.assertEqual(prepare_mock.call_count, 1)
            ui.button(key="clear_chat").click().run()
            self.assertEqual(len(ui.chat_message), 0)
            self.assertFalse(ui.chat_input[0].disabled)
            self.assertEqual(prepare_mock.call_count, 1)
            ui.number_input[0].set_value(90).run()
            self.assertTrue(ui.chat_input[0].disabled)

    def test_failed_preparation_does_not_enable_chat_or_expose_token(self):
        with patch("rag.application.prepare_assistant", side_effect=RuntimeError("failure hf_secret_test")):
            ui = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            ui.button(key="prepare").click().run()
            self.assertFalse(ui.exception)
            self.assertTrue(ui.chat_input[0].disabled)
            self.assertNotIn("hf_secret_test", " ".join(error.value for error in ui.error))

    def test_failed_forced_rebuild_resumes_without_forcing_again(self):
        fake = SimpleNamespace(cached_passages=1, close=Mock())
        with patch("rag.application.prepare_assistant", side_effect=[
            RuntimeError("Interrupted after one batch"), fake,
        ]) as prepare:
            ui = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            ui.checkbox(key="force_rebuild").check().run()
            ui.button(key="prepare").click().run()
            self.assertFalse(ui.exception)
            self.assertTrue(ui.chat_input[0].disabled)
            self.assertFalse(ui.checkbox(key="force_rebuild").value)
            ui.button(key="prepare").click().run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.chat_input[0].disabled)
            self.assertEqual([call.kwargs["force_rebuild"] for call in prepare.call_args_list],
                             [True, False])

    def test_chat_failure_keeps_index_ready_and_allows_retry(self):
        fake = SimpleNamespace(cached_passages=0, close=Mock(), answer=Mock(
            side_effect=RuntimeError("Model unavailable")))
        with patch("rag.application.prepare_assistant", return_value=fake) as prepare:
            ui = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            ui.button(key="prepare").click().run()
            ui.chat_input[0].set_value("Une question ?").run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.chat_input[0].disabled)
            self.assertIn("Model unavailable", ui.error[0].value)
            self.assertEqual(len(ui.session_state["messages"]), 0)
            fake.answer.side_effect = None
            fake.answer.return_value = {"answer": "Réponse.", "passages": [], "total_s": .1}
            ui.chat_input[0].set_value("Une question ?").run()
            self.assertFalse(ui.exception)
            self.assertEqual(len(ui.chat_message), 2)
            self.assertEqual(prepare.call_count, 1)


if __name__ == "__main__":
    unittest.main()
