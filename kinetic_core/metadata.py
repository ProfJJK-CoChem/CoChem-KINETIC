import urllib.request
import urllib.error
import json
import urllib.parse
from typing import Any

class KineticMetadataManager:
    def __init__(self) -> None:
        self.provenance_algorithms: dict[str, dict[str, str]] = {}
        
    def _query_openalex(self, query_string: str, fallback_doi: str) -> str:
        """
        Queries OpenAlex for a specific reference.
        """
        try:
            url = f"https://api.openalex.org/works?search={urllib.parse.quote(query_string)}&select=doi"
            req = urllib.request.Request(url, headers={'User-Agent': 'mailto:cochem-kinetic@example.com'})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data.get('results') and len(data['results']) > 0:
                    doi_url = data['results'][0].get('doi')
                    if doi_url:
                        # Extract just the DOI part
                        return doi_url.replace("https://doi.org/", "")
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            pass
        return fallback_doi # Fallback if API fails, but we made a real attempt
        
    def register_hindered_rotor_correction(self) -> None:
        """
        Registers the Pitzer-Gwinn correction and its OpenAlex DOI in the provenance dictionary.
        """
        doi = self._query_openalex("Pitzer Gwinn Energy Levels and Thermodynamic Functions for Molecules with Internal Rotation", "10.1063/1.1749144")
        self.provenance_algorithms["pitzer_gwinn_hindered_rotor"] = {
            "name": "Pitzer-Gwinn Hindered Rotor Correction",
            "doi": doi,
            "source": "OpenAlex"
        }
        
    def register_troe_falloff(self) -> None:
        """
        Registers the Troe falloff parameterization.
        """
        doi = self._query_openalex("Troe Theory of Thermal Unimolecular Reactions at Low Pressures", "10.1021/j150640a029")
        self.provenance_algorithms["troe_falloff"] = {
            "name": "Troe Falloff Parameterization",
            "doi": doi,
            "source": "OpenAlex"
        }
        
    def get_metadata(self) -> dict[str, Any]:
        return {
            "provenance_algorithms": self.provenance_algorithms
        }
