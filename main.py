import src.GRU_face as face
import src.GRU_pose as pose
import src.GRU_smile as smile
import src.GRU_late_fusion as LF_GRU
import src.GRU_late_fusion_game as LF_GRU_game

def main():
    # NOTE: Weight and Bias project name and your entitiy loggin name.
    PROJECT = 'Affective_Recognition'
    ENTITIY = 'iliancorneliussen'

    # NOTE: Uncomment the model that you would like to train.
    # face.main(project=PROJECT, entity=ENTITIY)
    # pose.main(project=PROJECT, entity=ENTITIY)
    # smile.main(project=PROJECT, entity=ENTITIY)
    # LF_GRU.main(project=PROJECT, entity=ENTITIY)
    LF_GRU_game.main(project=PROJECT, entity=ENTITIY)



if __name__ == '__main__':
    main()