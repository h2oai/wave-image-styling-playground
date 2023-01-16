from h2o_wave import ui


def get_layouts():
    layouts = [
        ui.layout(
            breakpoint="xs",
            zones=[
                ui.zone(
                    "header",
                    direction=ui.ZoneDirection.ROW,
                ),
                ui.zone(
                    "title",
                    direction=ui.ZoneDirection.ROW,
                ),
                ui.zone(
                    "main_wrapper",
                    direction=ui.ZoneDirection.ROW,
                    align="start",
                    zones=[
                        ui.zone(
                            "side_controls",
                            direction=ui.ZoneDirection.COLUMN,
                            size="20%",
                        ),
                        ui.zone(
                            "content",
                            size="80%",
                            direction=ui.ZoneDirection.COLUMN,
                            justify="center",
                            zones=[
                                ui.zone(
                                    "middle",
                                    direction=ui.ZoneDirection.ROW,
                                    justify="between",
                                    zones=[
                                        ui.zone(
                                            "middle_left",
                                            direction=ui.ZoneDirection.COLUMN,
                                            justify="center",
                                            align="center",
                                        ),
                                        ui.zone(
                                            "middle_right",
                                            direction=ui.ZoneDirection.COLUMN,
                                            justify="center",
                                            align="center",
                                        ),
                                    ],
                                ),
                                ui.zone(
                                    "main",
                                    direction=ui.ZoneDirection.ROW,
                                    justify="center",
                                    align="center",
                                ),
                                ui.zone(
                                    "bottom",
                                    direction=ui.ZoneDirection.ROW,
                                    justify="center",
                                    align="center",
                                ),
                            ],
                        ),
                    ],
                ),
                ui.zone("footer", direction=ui.ZoneDirection.ROW),
            ],
        )
    ]
    return layouts
