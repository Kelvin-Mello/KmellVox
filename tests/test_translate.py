"""Testes unitários para core/translate.py (Translator, batching, ModelProfile, VRAM cleanup)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hardware import ModelProfile
from core.transcribe import TranscriptionSegment
from core.translate import SYSTEM_PROMPT_TRANSLATION, TranslatedSegment, TranslationResult, Translator


class TestTranslator(unittest.TestCase):
    """Conjunto de testes para o módulo de tradução com LLM."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_filename_resolution(self):
        """Testa se o Translator seleciona o arquivo GGUF correto com base no ModelProfile."""
        # Perfil A
        prof_a = ModelProfile.from_profile("perfil_a")
        t_a = Translator(model_profile=prof_a, models_dir=self.temp_dir.name)
        self.assertEqual(t_a.model_filename, "Qwen3-8B-Instruct-Q4_K_M.gguf")
        self.assertIn("Qwen3-8B-Instruct-Q4_K_M.gguf", t_a.model_path)
        self.assertEqual(t_a.n_gpu_layers, -1)

        # Perfil B
        prof_b = ModelProfile.from_profile("perfil_b")
        t_b = Translator(model_profile=prof_b, models_dir=self.temp_dir.name)
        self.assertEqual(t_b.model_filename, "Qwen3-4B-Instruct-Q4_K_M.gguf")
        self.assertIn("Qwen3-4B-Instruct-Q4_K_M.gguf", t_b.model_path)
        self.assertEqual(t_b.n_gpu_layers, -1)

        # CPU
        prof_cpu = ModelProfile.from_profile("cpu")
        t_cpu = Translator(model_profile=prof_cpu, models_dir=self.temp_dir.name)
        self.assertEqual(t_cpu.model_filename, "Qwen3-1.5B-Instruct-Q4_K_M.gguf")
        self.assertEqual(t_cpu.n_gpu_layers, 0)

    def test_system_prompt_requirements(self):
        """Valida que o prompt de sistema contém diretrizes para nomes próprios, tom e sem comentários."""
        prompt = SYSTEM_PROMPT_TRANSLATION.lower()
        self.assertTrue("nomes próprios" in prompt or "nome" in prompt)
        self.assertTrue("tom" in prompt or "registro" in prompt)
        self.assertTrue("literal" in prompt)
        self.assertTrue("não adicione" in prompt or "sem comentários" in prompt or "comentários" in prompt)

    @patch("llama_cpp.Llama")
    def test_translate_segments_batch_and_vram_cleanup(self, mock_llama_cls):
        """Testa tradução em lotes (batch) com Llama mockado e validação de limpeza de VRAM."""
        mock_llm_instance = MagicMock()
        mock_llama_cls.return_value = mock_llm_instance

        # Simula resposta do LLM para um lote de 3 itens
        mock_llm_instance.return_value = {
            "choices": [
                {
                    "text": "[1] Olá pessoal, bem-vindos.\n[2] Hoje a Microsoft e a OpenAI anunciaram atualizações.\n[3] Vamos começar o tutorial."
                }
            ]
        }

        # Cria um arquivo mock de modelo GGUF para passar na verificação is_file()
        dummy_model_path = os.path.join(self.temp_dir.name, "Qwen3-8B-Instruct-Q4_K_M.gguf")
        with open(dummy_model_path, "w") as f:
            f.write("mock gguf")

        prof_a = ModelProfile.from_profile("perfil_a")
        translator = Translator(
            model_profile=prof_a,
            model_path=dummy_model_path,
        )

        segments = [
            TranscriptionSegment(id=1, start=0.0, end=2.0, text="Hello everyone, welcome."),
            TranscriptionSegment(id=2, start=2.2, end=5.5, text="Today Microsoft and OpenAI announced updates."),
            TranscriptionSegment(id=3, start=5.8, end=8.0, text="Let's start the tutorial."),
        ]

        translated = translator.translate_segments(
            segments=segments,
            source_language="en",
            target_language="pt",
            batch_size=3,
            auto_unload=True,
        )

        self.assertEqual(len(translated), 3)

        # Valida que os tempos originais foram rigorosamente preservados
        self.assertEqual(translated[0].start, 0.0)
        self.assertEqual(translated[0].end, 2.0)
        self.assertEqual(translated[0].original_text, "Hello everyone, welcome.")
        self.assertEqual(translated[0].translated_text, "Olá pessoal, bem-vindos.")

        self.assertEqual(translated[1].start, 2.2)
        self.assertEqual(translated[1].end, 5.5)
        self.assertEqual(translated[1].translated_text, "Hoje a Microsoft e a OpenAI anunciaram atualizações.")

        self.assertEqual(translated[2].start, 5.8)
        self.assertEqual(translated[2].end, 8.0)
        self.assertEqual(translated[2].translated_text, "Vamos começar o tutorial.")

        # Valida que o modelo foi liberado da VRAM
        self.assertIsNone(translator.llm)

    def test_translation_result_to_srt(self):
        """Testa geração de SRT a partir de TranslationResult."""
        segments = [
            TranslatedSegment(id=1, start=1.5, end=4.0, original_text="Hi", translated_text="Olá"),
            TranslatedSegment(id=2, start=4.5, end=7.2, original_text="World", translated_text="Mundo"),
        ]
        result = TranslationResult(source_language="en", target_language="pt", segments=segments)
        srt = result.to_srt()

        self.assertIn("1\n00:00:01,500 --> 00:00:04,000\nOlá", srt)
        self.assertIn("2\n00:00:04,500 --> 00:00:07,200\nMundo", srt)


if __name__ == "__main__":
    unittest.main()
