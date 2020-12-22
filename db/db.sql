create database matches;
create table matches.games
(
    id            bigint NOT NULL AUTO_INCREMENT,
    game_date     date         default null,
    first_player  varchar(255) DEFAULT NULL,
    second_player varchar(255) DEFAULT NULL,
    result        varchar(255) DEFAULT NULL,
    round         bigint not null,
    PRIMARY KEY (`id`)
)