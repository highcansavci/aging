import logging
from backend.adfd.service.aging_service import AgingService


class AgingController:

    @staticmethod
    def aging_task(np_arr):
        logging.info(
            "Executing aging task in the adfd model controller layer.")
        return AgingService.aging_task(np_arr)

    @staticmethod
    def check_face_alignment(np_arr):
        logging.info(
            "Checking face availability in the adfd model controller layer.")
        return AgingService.check_face_alignment(np_arr)
