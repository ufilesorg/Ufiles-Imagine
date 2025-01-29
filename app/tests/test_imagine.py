import logging
import os

import httpx
from server.config import Settings

Settings.config_logger()


def test_imagine():
    api_key = Settings.UFILES_API_KEY
    imagine_url = os.getenv(
        "IMAGINE_URL", "https://media.pixiee.io/v1/apps/imagine/imagination/bulk/"
    )

    with httpx.Client(headers={"x-api-key": api_key}) as client:
        response = client.post(
            imagine_url,
            params={
                "sync": True,
            },
            json={
                "delineation": "a beautiful red car",
                "aspect_ratios": ["3:2"],
                "engines": [
                    # "midjourney",
                    # "ideogram",
                    # "ideogram_turbo",
                    # "photon",
                    # "flux_1_1",
                    "flux_schnell",
                    # "photon_flash",
                    # "stability",
                    # "dalle",
                ],
            },
            timeout=None,
        )
        response.raise_for_status()
        logging.info(response.json())


if __name__ == "__main__":
    test_imagine()
