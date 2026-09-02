"""Módulo de tradução com IA utilizando llama-cpp-python e família Qwen (Qwen 3 GGUF)."""

from __future__ import annotations

import gc
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from core.hardware import ModelProfile

logger = logging.getLogger("KmellVox.Translate")

SYSTEM_PROMPT_TRANSLATION = """Você é um especialista sênior em tradução, localização e dublagem audiovisual.
Sua missão é traduzir o texto de fala do idioma de origem para o idioma de destino com altíssima qualidade e fidelidade.

Diretrizes obrigatórias:
1. Naturalidade e Fluidez Idiomática: Não faça tradução literal palavra por palavra. Use construções e vocabulário naturais na fala cotidiana do idioma de destino, adaptando expressões idiomáticas e gírias.
2. Preservação de Nomes Próprios: Mantenha nomes de pessoas, marcas, empresas, produtos, cidades e termos técnicos estabelecidos exatamente como no original.
3. Registro e Tom: Preserve fielmente o tom (formal, informal, coloquial, humorístico, emotivo ou técnico) do orador original.
4. Ritmo e Concisão: Adapte a extensão das frases para manter um tempo de pronúncia similar ao original, facilitando a sincronia labial.
5. Sem Comentários: NÃO adicione preâmbulos, explicações, notas de tradutor ou formatações extras. Retorne apenas e estritamente as falas traduzidas no formato solicitado.
"""


@dataclass
class TranslatedSegment:
    """Segmento traduzido preservando os timestamps originais."""
    id: int
    start: float
    end: float
    original_text: str
    translated_text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        """Permite acesso indexado como dicionário: seg['start'], seg['translated_text'], seg['text']."""
        if key == "text":
            return self.translated_text
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Chave '{key}' não encontrada no TranslatedSegment.")


@dataclass
class TranslationResult:
    """Resultado completo da tradução de uma lista de segmentos."""
    source_language: str
    target_language: str
    segments: List[TranslatedSegment] = field(default_factory=list)

    @property
    def full_translated_text(self) -> str:
        return " ".join(seg.translated_text.strip() for seg in self.segments)

    def to_srt(self) -> str:
        """Exporta a tradução no formato SRT."""
        def format_timestamp(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            if millis >= 1000:
                millis = 999
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        lines = []
        for i, seg in enumerate(self.segments, 1):
            start_str = format_timestamp(seg.start)
            end_str = format_timestamp(seg.end)
            lines.append(f"{i}\n{start_str} --> {end_str}\n{seg.translated_text.strip()}\n")
        return "\n".join(lines)

    def save_srt(self, filepath: str) -> str:
        """Salva a tradução como arquivo .srt."""
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_srt(), encoding="utf-8")
        return str(path)


class Translator:
    """
    Tradutor de falas para dublagem utilizando llama-cpp-python com modelos Qwen3 GGUF,
    processamento em lote (batch) e liberação explícita de VRAM.
    """

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        model_path: Optional[str] = None,
        models_dir: str = "models",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.3,
    ) -> None:
        """
        Inicializa o Translator.
        
        Args:
            model_profile: Perfil de hardware e modelos (se None, detecta automaticamente).
            model_path: Caminho explícito do arquivo .gguf (opcional).
            models_dir: Diretório raiz de modelos.
            n_ctx: Tamanho da janela de contexto.
            n_gpu_layers: Quantidade de camadas na GPU (-1 para todas).
            temperature: Temperatura de amostragem (baixa para traduções fiéis).
        """
        self.profile = model_profile or ModelProfile.from_profile()
        self.models_dir = Path(models_dir)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if self.profile.profile_name != "cpu" else 0
        self.temperature = temperature
        
        self.model_filename = self._resolve_model_filename()
        self.model_path = self._resolve_model_path(model_path)
        self.llm = None

    def _resolve_model_filename(self) -> str:
        """Determina o nome do arquivo .gguf com base no ModelProfile."""
        if hasattr(self.profile, "translation_filename") and self.profile.translation_filename:
            return self.profile.translation_filename
        if self.profile.profile_name == "perfil_a":
            return "Qwen3-8B-Instruct-Q4_K_M.gguf"
        elif self.profile.profile_name == "perfil_b":
            return "Qwen3-4B-Instruct-Q4_K_M.gguf"
        else:
            return "Qwen3-1.5B-Instruct-Q4_K_M.gguf"

    def _resolve_model_path(self, explicit_path: Optional[str]) -> Optional[str]:
        """Localiza o arquivo .gguf no disco."""
        if explicit_path and Path(explicit_path).is_file():
            return str(Path(explicit_path).resolve())

        # Busca em locais padrão do projeto
        candidates = [
            self.models_dir / "llm" / self.model_filename,
            self.models_dir / self.model_filename,
            self.models_dir / "llm" / self.model_filename.lower(),
            self.models_dir / self.model_filename.lower(),
            # Nomes alternativos Qwen 2.5
            self.models_dir / "llm" / "qwen2.5-7b-instruct-q4_k_m.gguf",
            self.models_dir / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        ]

        for cand in candidates:
            if cand.is_file():
                return str(cand.resolve())

        return str((self.models_dir / "llm" / self.model_filename).resolve())

    def load_model(self) -> None:
        """Carrega o modelo GGUF no llama-cpp-python."""
        if self.llm is not None:
            return

        if not self.model_path or not Path(self.model_path).is_file():
            logger.warning(
                "Arquivo do modelo LLM não encontrado em '%s'. "
                "Será utilizado modo fallback de tradução caso a inferência seja chamada.",
                self.model_path,
            )
            return

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError("llama-cpp-python não está instalado no ambiente virtual.") from e

        logger.info(
            "Carregando LLM GGUF [%s] no llama-cpp-python (n_ctx=%d, n_gpu_layers=%d, Perfil: %s)...",
            Path(self.model_path).name,
            self.n_ctx,
            self.n_gpu_layers,
            self.profile.profile_name,
        )

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        logger.info("LLM GGUF carregado com sucesso na memória.")

    def unload_model(self) -> None:
        """
        Libera explicitamente o modelo LLM da memória RAM e da VRAM da GPU (del + torch.cuda.empty_cache()),
        abrindo espaço livre para as etapas subsequentes do pipeline (TTS e Lip Sync).
        """
        if self.llm is not None:
            logger.info("Descarregando modelo LLM [%s] e liberando VRAM...", Path(self.model_path).name)
            del self.llm
            self.llm = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() executado com sucesso após LLM.")
        except Exception:
            pass

    def _translate_batch(
        self,
        batch_items: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """
        Traduz um lote de segmentos em uma única chamada de inferência LLM.
        """
        if not batch_items:
            return []

        if self.llm is None:
            self.load_model()

        # Fallback para desenvolvimento sem pesos baixados
        if self.llm is None:
            return [f"[{target_lang.upper()}] {item['text']}" for item in batch_items]

        # Monta prompt em lote
        items_text = "\n".join(f"[{item['id']}] {item['text'].strip()}" for item in batch_items)
        user_prompt = (
            f"Traduza os seguintes {len(batch_items)} segmentos de fala numerados de {source_lang} para {target_lang}.\n"
            f"Retorne cada tradução mantendo estritamente o identificador numérico correspondente '[ID] Tradução':\n\n"
            f"{items_text}"
        )

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT_TRANSLATION}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            response = self.llm(
                prompt,
                max_tokens=max(512, len(batch_items) * 128),
                temperature=self.temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            raw_text = response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error("Erro na inferência do llama-cpp-python: %s", e)
            return [item["text"] for item in batch_items]

        # Parseia as linhas retornadas [ID] Texto
        translated_map: Dict[int, str] = {}
        line_pattern = re.compile(r"^\[?(\d+)\]?[\.\:\-\)]?\s*(.+)$")

        for line in raw_text.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            match = line_pattern.match(line_s)
            if match:
                idx = int(match.group(1))
                text = match.group(2).strip().strip('"')
                translated_map[idx] = text

        # Mapeia de volta na ordem original com fallback individual se alguma chave faltar
        results: List[str] = []
        for item in batch_items:
            item_id = item["id"]
            if item_id in translated_map:
                results.append(translated_map[item_id])
            else:
                # Fallback: se o LLM omitiu um ID específico no batch
                logger.warning("Segmento [%d] não encontrado no retorno em lote do LLM. Usando texto original.", item_id)
                results.append(item["text"])

        return results

    def translate_segments(
        self,
        segments: List[Union[Dict[str, Any], Any]],
        target_language: str = "pt",
        source_language: str = "auto",
        batch_size: int = 8,
        auto_unload: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[TranslatedSegment]:
        """
        Traduz uma lista de segmentos de SRT em lotes (batch), preservando timestamps,
        idiomatismo e liberando VRAM ao final.
        
        Args:
            segments: Lista de segmentos (objetos com start, end, text ou dicts).
            target_language: Idioma de destino (ex: 'pt', 'es', 'en').
            source_language: Idioma de origem ou 'auto'.
            batch_size: Quantidade de segmentos por lote para inferência.
            auto_unload: Se True, descarrega o modelo da VRAM ao concluir.
            progress_callback: Callback (progresso_0_1, mensagem).
            
        Returns:
            List[TranslatedSegment]: Segmentos traduzidos com timestamps originais.
        """
        if not segments:
            return []

        if progress_callback:
            progress_callback(0.05, f"Iniciando tradução em lotes (tamanho={batch_size})...")

        # Normaliza segmentos de entrada
        normalized_items: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments, 1):
            if isinstance(seg, dict):
                s_id = int(seg.get("id", i))
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                text = str(seg.get("text", "")).strip()
            else:
                s_id = getattr(seg, "id", i)
                start = getattr(seg, "start", 0.0)
                end = getattr(seg, "end", 0.0)
                text = getattr(seg, "text", "").strip()

            normalized_items.append({
                "id": s_id,
                "start": start,
                "end": end,
                "text": text,
            })

        translated_segments: List[TranslatedSegment] = []
        total_items = len(normalized_items)

        try:
            # Itera em lotes
            for batch_start_idx in range(0, total_items, batch_size):
                batch = normalized_items[batch_start_idx : batch_start_idx + batch_size]
                current_num = batch_start_idx + len(batch)
                
                if progress_callback and total_items > 0:
                    prog = 0.05 + (batch_start_idx / total_items) * 0.90
                    progress_callback(prog, f"Traduzindo segmentos {batch_start_idx + 1}-{current_num} de {total_items}...")

                batch_translations = self._translate_batch(
                    batch_items=batch,
                    source_lang=source_language,
                    target_lang=target_language,
                )

                for item, trans_text in zip(batch, batch_translations):
                    translated_segments.append(
                        TranslatedSegment(
                            id=item["id"],
                            start=item["start"],
                            end=item["end"],
                            original_text=item["text"],
                            translated_text=trans_text,
                        )
                    )

            if progress_callback:
                progress_callback(1.0, "Tradução concluída com sucesso.")

            return translated_segments

        finally:
            if auto_unload:
                self.unload_model()


# Alias para compatibilidade retroativa
LLMTranslator = Translator
