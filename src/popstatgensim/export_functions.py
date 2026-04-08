"""Compatibility wrapper for export helpers moved into `popstatgensim.io`."""

from .io.gcta import export_GRM_GCTA, export_trait

__all__ = ["export_GRM_GCTA", "export_trait"]
